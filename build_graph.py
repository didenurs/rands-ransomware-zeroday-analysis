import os
import numpy as np
from scipy.sparse import load_npz, csr_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
K = 10


def p(name):
    return os.path.join(BASE_DIR, name)


print("1. Loading preprocessed data...")
X = load_npz(p("X_processed.npz")).tocsr().astype(np.float32)
y = np.load(p("y_labels.npy"))
y_binary = np.load(p("y_binary.npy"))
node_weights = np.load(p("node_weights.npy"))
n = X.shape[0]
print(f"-> Nodes: {n}, features: {X.shape[1]}")

print("2. Computing pairwise Jaccard similarity (binary features)...")
inter = (X @ X.T).toarray()
sizes = np.asarray(X.sum(axis=1)).ravel()
union = sizes[:, None] + sizes[None, :] - inter
with np.errstate(divide="ignore", invalid="ignore"):
    S = np.where(union > 0, inter / union, 0.0).astype(np.float32)
np.fill_diagonal(S, 0.0)

iu = np.triu_indices(n, k=1)
sims = S[iu]
q = np.quantile(sims, [0.5, 0.9, 0.99, 0.999])
print(f"-> Jaccard quantiles (50/90/99/99.9%): {np.round(q, 4)}")
print(f"-> Near-duplicate pairs (J >= 0.99): {(sims >= 0.99).sum()}")
print(f"-> Samples with no features: {(sizes == 0).sum()}")


def edge_homophily(rows, cols, labels):
    return float((labels[rows] == labels[cols]).mean())


def build_knn(S, k):
    """Directed top-k per row, then symmetrised (union, max weight)."""
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(S.shape[0]), k)
    cols = idx.ravel()
    w = S[rows, cols]
    keep = w > 0
    A = csr_matrix((w[keep], (rows[keep], cols[keep])), shape=S.shape)
    A = A.maximum(A.T).tocoo()
    return A.row.astype(np.int64), A.col.astype(np.int64), A.data.astype(np.float32)


print("3. k-NN comparison (undirected edges, family homophily):")
for k in (5, 10, 20):
    r, c, w = build_knn(S, k)
    iso = n - len(np.unique(r))
    print(f"   k={k:>2}: directed edges={len(r)}, isolated={iso}, "
          f"homophily(family)={edge_homophily(r, c, y):.3f}, "
          f"homophily(binary)={edge_homophily(r, c, y_binary):.3f}")

print(f"4. Building final graph with k={K}...")
row, col, weight = build_knn(S, K)
edge_index = np.vstack([row, col])
deg = np.bincount(row, minlength=n)
print(f"-> Edges (both directions): {edge_index.shape[1]}")
print(f"-> Degree min/mean/max: {deg.min()}/{deg.mean():.1f}/{deg.max()}")
print(f"-> Isolated nodes: {(deg == 0).sum()}")
print(f"-> Edge weight min/mean/max: {weight.min():.3f}/{weight.mean():.3f}/{weight.max():.3f}")
print(f"-> Edge homophily (family): {edge_homophily(row, col, y):.3f}")
print(f"-> Edge homophily (binary): {edge_homophily(row, col, y_binary):.3f}")

print("5. Saving graph (PyTorch Geometric layout)...")
np.save(p("graph_edge_index.npy"), edge_index)
np.save(p("graph_edge_weight.npy"), weight) 

try:
    import torch
    from torch_geometric.data import Data

    data = Data(
        x=torch.from_numpy(X.toarray()),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(weight),
        y=torch.from_numpy(y.astype(np.int64)),
        y_binary=torch.from_numpy(y_binary.astype(np.int64)),
        node_weight=torch.from_numpy(node_weights.astype(np.float32)),
    )
    torch.save(data, p("graph_pyg.pt"))
    print("-> Saved graph_pyg.pt (torch_geometric Data object)")
except ImportError:
    print("-> torch / torch_geometric not installed: saved .npy files only.")

print("\n--- GRAPH CONSTRUCTION COMPLETED ---")
