import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from config import (
    DATA_DIR, OUTPUT_DIR, SPLIT_SCENARIO1_CSV, SPLIT_SCENARIO2_CSV,
    MAX_FEATURES, SUBLINEAR_TF, RANDOM_SEED
)

TOKENS_CACHE = os.path.join(DATA_DIR, "tokens_cache.parquet")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------- Deep Learning PyTorch Models -----------------
class DeepMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

class Conv1DNet(nn.Module):
    def __init__(self, seq_len=128):
        super().__init__()
        # Input shape: (batch, 1, seq_len)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(32 * (seq_len // 2), 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        # x is (batch, seq_len) -> unsqueeze to (batch, 1, seq_len)
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = x.flatten(1)
        return self.fc(x)

class BiLSTMNet(nn.Module):
    def __init__(self, seq_len=16, feat_dim=8):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.lstm = nn.LSTM(feat_dim, 32, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        # x is (batch, 128) -> reshape to (batch, 16, 8)
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.feat_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        return self.fc(out)

def train_pytorch_model(model, train_loader, val_loader, epochs=15, lr=0.001, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_loss = float('inf')
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                val_loss += criterion(out, by).item() * bx.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_weights:
        model.load_state_dict(best_weights)
    return model

def evaluate_predictions(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    pr_auc = average_precision_score(y_true, y_prob) if y_prob is not None else 0.0
    
    return {
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'Accuracy': float(acc), 'Precision': float(prec), 'Recall': float(rec),
        'F1_Score': float(f1), 'FPR': float(fpr), 'ROC_AUC': float(roc_auc),
        'PR_AUC': float(pr_auc)
    }

def run_experiment_for_scenario(scenario_name, split_csv_path, df_tokens):
    print(f"\n=======================================================")
    print(f"       Running Experiment: {scenario_name}")
    print(f"=======================================================")
    
    df_split = pd.read_csv(split_csv_path)
    # Merge with tokens cache on SHA256
    merged = pd.merge(df_split, df_tokens, on='SHA256', how='inner')
    print(f"Merged samples count: {len(merged)}")
    
    train_mask = (merged['Split'] == 'train').values
    val_mask = (merged['Split'] == 'val').values
    test_mask = (merged['Split'] == 'test').values
    
    X_train_raw = merged.loc[train_mask, 'Tokens'].tolist()
    y_train = merged.loc[train_mask, 'Label'].values
    
    X_val_raw = merged.loc[val_mask, 'Tokens'].tolist()
    y_val = merged.loc[val_mask, 'Label'].values
    
    X_test_raw = merged.loc[test_mask, 'Tokens'].tolist()
    y_test = merged.loc[test_mask, 'Label'].values
    
    print(f"Dataset sizes -> Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")
    
    print("\nFitting TF-IDF Vectorizer ONLY on Train split (Zero Leakage)...")
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES, sublinear_tf=SUBLINEAR_TF, token_pattern=r'\S+')
    t0 = time.time()
    X_train_tfidf = tfidf.fit_transform(X_train_raw)
    t1 = time.time()
    print(f"TF-IDF fit_transform on Train completed in {t1 - t0:.2f}s. Shape: {X_train_tfidf.shape}")
    
    print("Transforming Val and Test splits...")
    X_val_tfidf = tfidf.transform(X_val_raw)
    X_test_tfidf = tfidf.transform(X_test_raw)
    
    # Save vectorizer
    joblib.dump(tfidf, os.path.join(MODELS_DIR, f"tfidf_{scenario_name}.joblib"))
    
    # Precompute SVD for KNN and DL
    print("Fitting TruncatedSVD(128) on Train for KNN and DL sequences...")
    svd = TruncatedSVD(n_components=128, random_state=RANDOM_SEED)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_val_svd = svd.transform(X_val_tfidf)
    X_test_svd = svd.transform(X_test_tfidf)
    
    scenario_results = {}
    
    # 1. Logistic Regression (No Val)
    print("\n--- Training 1. Logistic Regression ---")
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED, n_jobs=-1)
    lr.fit(X_train_tfidf, y_train)
    t1 = time.time()
    y_pred = lr.predict(X_test_tfidf)
    y_prob = lr.predict_proba(X_test_tfidf)[:, 1]
    res_lr = evaluate_predictions(y_test, y_pred, y_prob)
    res_lr['Train_Time_s'] = round(t1 - t0, 2)
    scenario_results['Logistic Regression'] = res_lr
    print("LR Results:", res_lr)
    joblib.dump(lr, os.path.join(MODELS_DIR, f"lr_{scenario_name}.joblib"))
    
    # 2. Linear SVM with calibration (No Val)
    print("\n--- Training 2. Support Vector Machine (LinearSVC) ---")
    t0 = time.time()
    base_svc = LinearSVC(C=1.0, dual=False, max_iter=2000, random_state=RANDOM_SEED)
    cal_svc = CalibratedClassifierCV(estimator=base_svc, cv=3)
    cal_svc.fit(X_train_tfidf, y_train)
    t1 = time.time()
    y_pred = cal_svc.predict(X_test_tfidf)
    y_prob = cal_svc.predict_proba(X_test_tfidf)[:, 1]
    res_svm = evaluate_predictions(y_test, y_pred, y_prob)
    res_svm['Train_Time_s'] = round(t1 - t0, 2)
    scenario_results['SVM (Linear)'] = res_svm
    print("SVM Results:", res_svm)
    joblib.dump(cal_svc, os.path.join(MODELS_DIR, f"svm_{scenario_name}.joblib"))
    
    # 3. Random Forest (No Val)
    print("\n--- Training 3. Random Forest ---")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=25, n_jobs=6, random_state=RANDOM_SEED)
    rf.fit(X_train_tfidf, y_train)
    t1 = time.time()
    y_pred = rf.predict(X_test_tfidf)
    y_prob = rf.predict_proba(X_test_tfidf)[:, 1]
    res_rf = evaluate_predictions(y_test, y_pred, y_prob)
    res_rf['Train_Time_s'] = round(t1 - t0, 2)
    scenario_results['Random Forest'] = res_rf
    print("RF Results:", res_rf)
    joblib.dump(rf, os.path.join(MODELS_DIR, f"rf_{scenario_name}.joblib"))
    
    # 4. K-Nearest Neighbors (using SVD representation for scalability) (No Val)
    print("\n--- Training 4. K-Nearest Neighbors (KNN) ---")
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=6)
    knn.fit(X_train_svd, y_train)
    t1 = time.time()
    y_pred = knn.predict(X_test_svd)
    y_prob = knn.predict_proba(X_test_svd)[:, 1]
    res_knn = evaluate_predictions(y_test, y_pred, y_prob)
    res_knn['Train_Time_s'] = round(t1 - t0, 2)
    scenario_results['KNN'] = res_knn
    print("KNN Results:", res_knn)
    joblib.dump(knn, os.path.join(MODELS_DIR, f"knn_{scenario_name}.joblib"))
    
    # 5. XGBoost (Uses Val for Early Stopping)
    print("\n--- Training 5. XGBoost (with Validation Early Stopping) ---")
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=250, learning_rate=0.08, max_depth=6,
        tree_method='hist', early_stopping_rounds=15,
        eval_metric='logloss', random_state=RANDOM_SEED, n_jobs=6
    )
    xgb_model.fit(X_train_tfidf, y_train, eval_set=[(X_val_tfidf, y_val)], verbose=False)
    t1 = time.time()
    y_pred = xgb_model.predict(X_test_tfidf)
    y_prob = xgb_model.predict_proba(X_test_tfidf)[:, 1]
    res_xgb = evaluate_predictions(y_test, y_pred, y_prob)
    res_xgb['Train_Time_s'] = round(t1 - t0, 2)
    scenario_results['XGBoost'] = res_xgb
    print("XGBoost Results:", res_xgb)
    xgb_model.save_model(os.path.join(MODELS_DIR, f"xgb_{scenario_name}.json"))
    
    # Prepare PyTorch Tensors
    train_ds_svd = TensorDataset(torch.tensor(X_train_svd, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds_svd = TensorDataset(torch.tensor(X_val_svd, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds_svd = TensorDataset(torch.tensor(X_test_svd, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_ds_svd, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds_svd, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds_svd, batch_size=512, shuffle=False)
    
    # 6. Deep MLP (Uses Val for Early Stopping)
    print("\n--- Training 6. Deep MLP (PyTorch, with Val Early Stopping) ---")
    t0 = time.time()
    mlp = DeepMLP(input_dim=128)
    mlp = train_pytorch_model(mlp, train_loader, val_loader, epochs=12, lr=0.002)
    t1 = time.time()
    mlp.eval()
    with torch.no_grad():
        preds, probs = [], []
        for bx, _ in test_loader:
            out = mlp(bx)
            pr = torch.softmax(out, dim=1)
            preds.extend(pr.argmax(dim=1).numpy())
            probs.extend(pr[:, 1].numpy())
    res_mlp = evaluate_predictions(y_test, np.array(preds), np.array(probs))
    res_mlp['Train_Time_s'] = round(t1 - t0, 2)
    scenario_results['Deep MLP'] = res_mlp
    print("Deep MLP Results:", res_mlp)
    torch.save(mlp.state_dict(), os.path.join(MODELS_DIR, f"mlp_{scenario_name}.pt"))
    
    # 7. 1D-CNN (Uses Val for Early Stopping)
    print("\n--- Training 7. 1D-CNN (PyTorch, with Val Early Stopping) ---")
    t0 = time.time()
    cnn = Conv1DNet(seq_len=128)
    cnn = train_pytorch_model(cnn, train_loader, val_loader, epochs=12, lr=0.001)
    t1 = time.time()
    cnn.eval()
    with torch.no_grad():
        preds, probs = [], []
        for bx, _ in test_loader:
            out = cnn(bx)
            pr = torch.softmax(out, dim=1)
            preds.extend(pr.argmax(dim=1).numpy())
            probs.extend(pr[:, 1].numpy())
    res_cnn = evaluate_predictions(y_test, np.array(preds), np.array(probs))
    res_cnn['Train_Time_s'] = round(t1 - t0, 2)
    scenario_results['1D-CNN'] = res_cnn
    print("1D-CNN Results:", res_cnn)
    torch.save(cnn.state_dict(), os.path.join(MODELS_DIR, f"cnn_{scenario_name}.pt"))
    
    # 8. BiLSTM (Uses Val for Early Stopping)
    print("\n--- Training 8. Bi-LSTM (PyTorch, with Val Early Stopping) ---")
    t0 = time.time()
    lstm = BiLSTMNet(seq_len=16, feat_dim=8)
    lstm = train_pytorch_model(lstm, train_loader, val_loader, epochs=12, lr=0.002)
    t1 = time.time()
    lstm.eval()
    with torch.no_grad():
        preds, probs = [], []
        for bx, _ in test_loader:
            out = lstm(bx)
            pr = torch.softmax(out, dim=1)
            preds.extend(pr.argmax(dim=1).numpy())
            probs.extend(pr[:, 1].numpy())
    res_lstm = evaluate_predictions(y_test, np.array(preds), np.array(probs))
    res_lstm['Train_Time_s'] = round(t1 - t0, 2)
    scenario_results['Bi-LSTM'] = res_lstm
    print("Bi-LSTM Results:", res_lstm)
    torch.save(lstm.state_dict(), os.path.join(MODELS_DIR, f"bilstm_{scenario_name}.pt"))
    
    # Save scenario results
    res_path = os.path.join(OUTPUT_DIR, f"results_{scenario_name}.json")
    with open(res_path, 'w') as f:
        json.dump(scenario_results, f, indent=4)
    print(f"\nSaved {scenario_name} results to {res_path}")
    
    return scenario_results

def main():
    print(f"Loading tokens cache from {TOKENS_CACHE}...")
    df_tokens = pd.read_parquet(TOKENS_CACHE)
    print(f"Loaded {len(df_tokens)} tokenized samples.")
    
    res1 = run_experiment_for_scenario("Scenario_1_Known_Family", SPLIT_SCENARIO1_CSV, df_tokens)
    res2 = run_experiment_for_scenario("Scenario_2_Zero_Day", SPLIT_SCENARIO2_CSV, df_tokens)
    
    # Combine into a unified comparison
    comparison = {
        'Scenario_1_Known_Family': res1,
        'Scenario_2_Zero_Day': res2
    }
    
    comp_json_path = os.path.join(OUTPUT_DIR, "all_experiments_comparison.json")
    with open(comp_json_path, 'w') as f:
        json.dump(comparison, f, indent=4)
        
    # Also save as readable CSV
    rows = []
    for sc_name, models_dict in comparison.items():
        for m_name, m_res in models_dict.items():
            row = {'Scenario': sc_name, 'Model': m_name}
            row.update(m_res)
            rows.append(row)
    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(os.path.join(OUTPUT_DIR, "all_experiments_comparison.csv"), index=False)
    print(f"\nSuccessfully saved full comparison to {comp_json_path} and CSV!")

if __name__ == '__main__':
    main()
