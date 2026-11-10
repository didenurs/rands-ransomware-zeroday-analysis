import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from scipy.sparse import csr_matrix, save_npz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "RansomwareData.csv")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "data", "feature_names.csv")

if not os.path.exists(DATA_PATH) and os.path.exists("RansomwareData.csv"):
    DATA_PATH = "RansomwareData.csv"

print("1. Loading dataset...")
df = pd.read_csv(DATA_PATH, header=None)
print(f"-> Dataset shape: {df.shape[0]} samples, {df.shape[1]} columns")

feature_names = pd.read_csv(FEATURE_NAMES_PATH, header=None)[0].values
sample_ids = df.iloc[:, 0].values
y_binary = df.iloc[:, 1].values.astype(int)
y_family = df.iloc[:, 2].values.astype(int)
X_df = df.iloc[:, 3:]

if X_df.shape[1] != len(feature_names):
    print(f"-> WARNING: {X_df.shape[1]} feature columns vs {len(feature_names)} feature names")

classes, y_encoded = np.unique(y_family, return_inverse=True)
print(f"-> Target: family column (col 2), {len(classes)} classes")

if X_df.isnull().sum().sum() > 0:
    print("-> Missing values detected. Imputing with zeros...")
    X_df = X_df.fillna(0)

X = X_df.values.astype(np.float32)

print("2. Removing zero-variance features...")
selector = VarianceThreshold(threshold=0.0)
X_clean = selector.fit_transform(X)
print(f"-> Feature dimension: {X.shape[1]} -> reduced to {X_clean.shape[1]}")

node_activity = np.sum(X_clean, axis=1)
node_weights = node_activity / np.max(node_activity)

X_sparse = csr_matrix(X_clean)

print("3. Saving preprocessed artifacts...")
save_npz("X_processed.npz", X_sparse)
np.save("y_labels.npy", y_encoded)
np.save("node_weights.npy", node_weights)
np.save("classes.npy", classes)
np.save("y_binary.npy", y_binary)
np.save("sample_ids.npy", sample_ids)
np.save("feature_names_selected.npy", feature_names[selector.get_support()[:len(feature_names)]] if len(feature_names) == X.shape[1] else feature_names)

print("\n--- PREPROCESSING COMPLETED SUCCESSFULLY ---")
print(f"Total Nodes: {X_clean.shape[0]}")
print(f"Node Feature Dimension: {X_clean.shape[1]}")
print(f"Class Distribution: {dict(zip(classes, np.bincount(y_encoded)))}")