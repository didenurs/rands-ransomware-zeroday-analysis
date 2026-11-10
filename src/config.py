import os

# Project Root Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_new")
ZIP_PATH = os.path.join(DATA_DIR, "RanDS_Behaviour_Activity_Dataset.zip")
ZIP_PASSWORD = b"infected"

# Metadata Paths
RANSOMWARE_CSV = os.path.join(DATA_DIR, "Ransomware.csv")
BENIGN_CSV = os.path.join(DATA_DIR, "Benign.csv")

# Processed Split Manifests
SPLIT_SCENARIO1_CSV = os.path.join(DATA_DIR, "split_scenario1.csv")
SPLIT_SCENARIO2_CSV = os.path.join(DATA_DIR, "split_scenario2.csv")

# Extracted Features and Models Output
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# TF-IDF Settings
MAX_FEATURES = 15000
SUBLINEAR_TF = True
