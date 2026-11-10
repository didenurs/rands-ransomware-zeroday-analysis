import os
import json
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import load_npz
from scipy.sparse.csgraph import connected_components
from torch_geometric.nn import GCNConv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]
DUP_THRESHOLD = 0.95
KS = [5, 10]
EPOCHS = 120


def p(name):
    return os.path.join(BASE_DIR, name)


print("1. Loading datasets...")
X_sp = load_npz(p("X_processed.npz")).tocsr().astype(np.float32)
y_bin = np.load(p("y_binary.npy")).astype(np.int64)
y_fam = np.load(p("y_labels.npy")).astype(np.int64)
node_w = np.load(p("node_weights.npy")).astype(np.float32)

N, D = X_sp.shape
print(f"-> Samples: {N}, features: {D}, benign: {(y_bin == 0).sum()}, ransomware: {(y_bin == 1).sum()}")

print("2. Computing pairwise Jaccard similarity...")
inter = (X_sp @ X_sp.T).toarray()
sizes = np.asarray(X_sp.sum(axis=1)).ravel()
union = sizes[:, None] + sizes[None, :] - inter
S = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0).astype(np.float32)
np.fill_diagonal(S, 0.0)
del inter, union

print("3. Connected components at J >= 0.95...")
n_groups, group = connected_components(S >= DUP_THRESHOLD, directed=False)
group_sizes = np.bincount(group)
print(f"-> Found {n_groups} groups (max size: {group_sizes.max()})")

X_aug = np.hstack([X_sp.toarray(), node_w[:, None]])
x_all = torch.tensor(X_aug, dtype=torch.float32)
y_all = torch.tensor(y_bin)
del X_aug


def group_split(rng, train_r=0.70, val_r=0.15):
    g_label = np.array([int(np.round(y_bin[group == g].mean())) for g in range(n_groups)])
    parts = {"train": [], "val": [], "test": []}
    for lab in (0, 1):
        gs = np.where(g_label == lab)[0]
        rng.shuffle(gs)
        total = group_sizes[gs].sum()
        acc = 0
        for g in gs:
            if acc < total * train_r:
                parts["train"].append(g)
            elif acc < total * (train_r + val_r):
                parts["val"].append(g)
            else:
                parts["test"].append(g)
            acc += group_sizes[g]
    return [np.where(np.isin(group, parts[k]))[0] for k in ("train", "val", "test")]


def knn_edges(src_nodes, dst_nodes, k, exclude_self=False, symmetric=False):
    sub = S[np.ix_(dst_nodes, src_nodes)].copy()
    if exclude_self:
        np.fill_diagonal(sub, -1.0)
    k = min(k, sub.shape[1] - 1)
    top = np.argpartition(-sub, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(len(dst_nodes)), k)
    cols = top.ravel()
    w = sub[rows, cols]
    keep = w > 0
    s = src_nodes[cols[keep]]
    d = dst_nodes[rows[keep]]
    w = w[keep]
    if symmetric:
        s, d, w = np.concatenate([s, d]), np.concatenate([d, s]), np.concatenate([w, w])
    return (torch.tensor(np.vstack([s, d]), dtype=torch.long),
            torch.tensor(w, dtype=torch.float32))


def compute_metrics(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tp = np.sum((pred == 1) & (y_true == 1))
    tn = np.sum((pred == 0) & (y_true == 0))
    fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    fpr = fp / max(fp + tn, 1)

    order = np.argsort(-prob, kind="stable")
    ys = y_true[order]
    ps = prob[order]
    distinct = np.r_[np.where(np.diff(ps))[0], len(ps) - 1]
    tps = np.cumsum(ys)[distinct]
    fps = 1 + distinct - tps
    P, Nn = max(ys.sum(), 1), max(len(ys) - ys.sum(), 1)
    tpr_c = np.r_[0.0, tps / P]
    fpr_c = np.r_[0.0, fps / Nn]
    roc = float(np.sum(np.diff(fpr_c) * (tpr_c[1:] + tpr_c[:-1]) / 2))
    prec_c = tps / (tps + fps)
    rec_c = tps / P
    pr = float(np.sum(np.diff(np.r_[0.0, rec_c]) * prec_c))
    return {"Precision": prec, "Recall": rec, "F1": f1, "ROC-AUC": roc, "PR-AUC": pr, "FPR": fpr}


class MLP(nn.Module):
    def __init__(self, d, h=128, dropout=0.4):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, 2)
        self.dp = dropout

    def forward(self, x, ei=None, ew=None):
        x = F.dropout(F.relu(self.fc1(x)), self.dp, self.training)
        return self.fc2(x)


class GCN(nn.Module):
    def __init__(self, d, h=128, dropout=0.4):
        super().__init__()
        self.c1 = GCNConv(d, h)
        self.c2 = GCNConv(h, 2)
        self.dp = dropout

    def forward(self, x, ei, ew):
        x = F.dropout(F.relu(self.c1(x, ei, edge_weight=ew)), self.dp, self.training)
        return self.c2(x, ei, edge_weight=ew)


def train_and_eval(model, g_train, g_val, g_test, tr, va, te, lr=1e-3, wd=5e-4):
    n_b, n_r = max((y_bin[tr] == 0).sum(), 1), max((y_bin[tr] == 1).sum(), 1)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, n_b / n_r], dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_score, best_state = -1.0, None

    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = crit(model(x_all, *g_train)[tr], y_all[tr])
        loss.backward()
        opt.step()

        if ep % 2 == 0 or ep == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                pv = F.softmax(model(x_all, *g_val), 1)[va, 1].numpy()
            m = compute_metrics(y_bin[va], pv)
            score = m["F1"] + m["PR-AUC"]
            if score > best_score:
                best_score = score
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pt = F.softmax(model(x_all, *g_test), 1)[te, 1].numpy()
    return compute_metrics(y_bin[te], pt)


print("\n=======================================================")
print("PROTOCOL A: UNSEEN SAMPLE (5 SEEDS, LEAKAGE-FREE)")
print("=======================================================")
names = ["MLP", "GCN (k=5)", "GCN (k=10)"]
sample_res = {n: [] for n in names}

for seed in SEEDS:
    rng = np.random.RandomState(seed)
    tr, va, te = group_split(rng)
    max_leak = S[np.ix_(te, tr)].max()
    print(f"Seed {seed}: train={len(tr)}, val={len(va)}, test={len(te)}, max test-train Jaccard={max_leak:.4f}")
    assert max_leak < DUP_THRESHOLD

    torch.manual_seed(seed)
    none_g = (None, None)
    sample_res["MLP"].append(train_and_eval(MLP(x_all.shape[1]), none_g, none_g, none_g, tr, va, te))

    for k in KS:
        g_tr = knn_edges(tr, tr, k, exclude_self=True, symmetric=True)
        ev = knn_edges(tr, va, k)
        et = knn_edges(tr, te, k)
        g_va = (torch.cat([g_tr[0], ev[0]], 1), torch.cat([g_tr[1], ev[1]]))
        g_te = (torch.cat([g_tr[0], et[0]], 1), torch.cat([g_tr[1], et[1]]))
        torch.manual_seed(seed)
        sample_res[f"GCN (k={k})"].append(train_and_eval(GCN(x_all.shape[1]), g_tr, g_va, g_te, tr, va, te))

print("\n=======================================================")
print("PROTOCOL B: UNSEEN FAMILY (FAMILY-HELD-OUT 4-FOLD)")
print("=======================================================")
family_folds = [
    [2, 8],        # fold 0: 107 + 4 = 111 samples
    [6, 10, 4],    # fold 1: 97 + 6 + 25 = 128 samples
    [9, 1, 11],    # fold 2: 90 + 50 + 34 = 174 samples
    [5, 7, 3]      # fold 3: 64 + 59 + 46 = 169 samples
]

benign_groups = [g for g in range(n_groups) if int(np.round(y_bin[group == g].mean())) == 0]
rng_fam = np.random.RandomState(42)
rng_fam.shuffle(benign_groups)
b_split = np.array_split(benign_groups, len(family_folds))

fam_res = {n: [] for n in names}

for fold_idx, test_fams in enumerate(family_folds):
    te_r_idx = np.where(np.isin(y_fam, test_fams))[0]
    te_b_groups = b_split[fold_idx]
    te_b_idx = np.where(np.isin(group, te_b_groups))[0]
    te = np.concatenate([te_r_idx, te_b_idx])

    rem_fams = [f for f in range(1, 12) if f not in test_fams]
    all_train_r_idx = np.where(np.isin(y_fam, rem_fams))[0]

    rem_b_groups = [g for i, b in enumerate(b_split) if i != fold_idx for g in b]
    n_rem_b = len(rem_b_groups)
    tr_b_groups = rem_b_groups[:int(n_rem_b * 0.8)]
    va_b_groups = rem_b_groups[int(n_rem_b * 0.8):]

    tr_b_idx = np.where(np.isin(group, tr_b_groups))[0]
    va_b_idx = np.where(np.isin(group, va_b_groups))[0]

    rng_f = np.random.RandomState(fold_idx)
    perm_r = rng_f.permutation(all_train_r_idx)
    n_r_tr = int(len(perm_r) * 0.85)
    tr_r_idx = perm_r[:n_r_tr]
    va_r_idx = perm_r[n_r_tr:]

    tr = np.concatenate([tr_b_idx, tr_r_idx])
    va = np.concatenate([va_b_idx, va_r_idx])

    max_leak_b = S[np.ix_(te_b_idx, tr_b_idx)].max() if len(te_b_idx) > 0 else 0.0
    print(f"Fold {fold_idx}: Test fams={test_fams} (N={len(te_r_idx)}), Benign test={len(te_b_idx)}, Train={len(tr)}, Val={len(va)}")

    torch.manual_seed(fold_idx)
    none_g = (None, None)
    fam_res["MLP"].append(train_and_eval(MLP(x_all.shape[1]), none_g, none_g, none_g, tr, va, te))

    for k in KS:
        g_tr = knn_edges(tr, tr, k, exclude_self=True, symmetric=True)
        ev = knn_edges(tr, va, k)
        et = knn_edges(tr, te, k)
        g_va = (torch.cat([g_tr[0], ev[0]], 1), torch.cat([g_tr[1], ev[1]]))
        g_te = (torch.cat([g_tr[0], et[0]], 1), torch.cat([g_tr[1], et[1]]))
        torch.manual_seed(fold_idx)
        fam_res[f"GCN (k={k})"].append(train_and_eval(GCN(x_all.shape[1]), g_tr, g_va, g_te, tr, va, te))


keys = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR"]

out_lines = []
out_lines.append("=" * 110)
out_lines.append("RESULTS SUMMARY: PROTOCOL A (UNSEEN SAMPLE - 5 SEEDS)")
out_lines.append("=" * 110)
header = f"{'Model':<12}" + "".join(f"| {k:<15}" for k in keys)
out_lines.append(header)
out_lines.append("-" * 110)

for n in names:
    row = f"{n:<12}"
    for k in keys:
        v = np.array([r[k] for r in sample_res[n]])
        row += f"| {v.mean():.4f}+-{v.std():.4f} "
    out_lines.append(row)

out_lines.append("\n" + "=" * 110)
out_lines.append("STATISTICAL SIGNIFICANCE (PAIRED T-TEST ON 5 SEEDS)")
out_lines.append("=" * 110)
for pair_name, m1, m2 in [("GCN(k=5) vs GCN(k=10)", "GCN (k=5)", "GCN (k=10)"),
                           ("MLP vs GCN(k=5)", "MLP", "GCN (k=5)")]:
    out_lines.append(f"\nComparison: {pair_name}")
    for k in ["F1", "ROC-AUC", "PR-AUC", "FPR"]:
        v1 = np.array([r[k] for r in sample_res[m1]])
        v2 = np.array([r[k] for r in sample_res[m2]])
        diff = v1 - v2
        t_stat, p_val = stats.ttest_rel(v1, v2)
        out_lines.append(f"  {k:<8}: mean diff={diff.mean():+.4f}, t={t_stat:+.3f}, p-val={p_val:.4f}")

out_lines.append("\n" + "=" * 110)
out_lines.append("RESULTS SUMMARY: PROTOCOL B (UNSEEN FAMILY - 4 FOLDS)")
out_lines.append("=" * 110)
out_lines.append(header)
out_lines.append("-" * 110)

for n in names:
    row = f"{n:<12}"
    for k in keys:
        v = np.array([r[k] for r in fam_res[n]])
        row += f"| {v.mean():.4f}+-{v.std():.4f} "
    out_lines.append(row)

print("\n".join(out_lines))

with open(p("results.txt"), "w") as f:
    f.write("\n".join(out_lines) + "\n")

all_data = {
    "unseen_sample": sample_res,
    "unseen_family": fam_res
}
with open(p("detailed_results.json"), "w") as f:
    json.dump(all_data, f, indent=2)

print("\nSaved detailed results to results.txt and detailed_results.json")
