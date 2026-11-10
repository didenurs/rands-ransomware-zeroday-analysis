import os
import pandas as pd
import numpy as np
from config import (
    RANSOMWARE_CSV, BENIGN_CSV, SPLIT_SCENARIO1_CSV, SPLIT_SCENARIO2_CSV,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)

def load_and_clean_metadata():
    print("Loading metadata CSVs...")
    df_ran = pd.read_csv(RANSOMWARE_CSV)
    df_ben = pd.read_csv(BENIGN_CSV)
    
    # In RanDS Ransomware.csv, 'Entropy' column contains the true family names
    # and 'Family' was a shifted binary column. We rename and clean.
    df_ran_clean = pd.DataFrame({
        'SHA256': df_ran['SHA256'].astype(str),
        'Label': 1,
        'Family': df_ran['Entropy'].astype(str),
        'Filepath': df_ran['Filepath'].astype(str)
    })
    
    df_ben_clean = pd.DataFrame({
        'SHA256': df_ben['SHA256'].astype(str),
        'Label': 0,
        'Family': 'Benign',
        'Filepath': df_ben['Filepath'].astype(str)
    })
    
    print(f"Loaded {len(df_ran_clean)} Ransomware samples across {df_ran_clean['Family'].nunique()} families.")
    print(f"Loaded {len(df_ben_clean)} Benign samples.")
    return df_ran_clean, df_ben_clean

def split_benign(df_ben, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(df_ben))
    n_train = int(np.round(TRAIN_RATIO * len(df_ben)))
    n_val = int(np.round(VAL_RATIO * len(df_ben)))
    
    splits = np.empty(len(df_ben), dtype=object)
    splits[shuffled_indices[:n_train]] = 'train'
    splits[shuffled_indices[n_train:n_train + n_val]] = 'val'
    splits[shuffled_indices[n_train + n_val:]] = 'test'
    
    df_res = df_ben.copy()
    df_res['Split'] = splits
    return df_res

def generate_scenario1(df_ran, df_ben_split, seed=RANDOM_SEED):
    """
    Scenario 1: Random 70% Train, 15% Val, 15% Test within EACH ransomware family.
    """
    print("\n--- Generating Scenario 1 (Known Family Split) ---")
    rng = np.random.default_rng(seed)
    ran_splits = np.empty(len(df_ran), dtype=object)
    
    for family, group in df_ran.groupby('Family'):
        indices = group.index.to_numpy(copy=True)
        n_fam = len(indices)
        rng.shuffle(indices)
        
        if n_fam == 1:
            # 1 sample -> train
            ran_splits[indices[0]] = 'train'
        elif n_fam == 2:
            ran_splits[indices[0]] = 'train'
            # allocate second sample randomly between val and test
            ran_splits[indices[1]] = rng.choice(['val', 'test'])
        else:
            n_tr = int(np.round(TRAIN_RATIO * n_fam))
            n_va = int(np.round(VAL_RATIO * n_fam))
            # ensure at least 1 in val/test if n_fam >= 4
            if n_tr >= n_fam:
                n_tr = n_fam - 2 if n_fam >= 3 else n_fam - 1
            if n_va >= n_fam - n_tr and n_fam - n_tr > 1:
                n_va = n_fam - n_tr - 1
            
            n_te = n_fam - n_tr - n_va
            ran_splits[indices[:n_tr]] = 'train'
            ran_splits[indices[n_tr:n_tr + n_va]] = 'val'
            ran_splits[indices[n_tr + n_va:]] = 'test'
            
    df_ran_split = df_ran.copy()
    df_ran_split['Split'] = ran_splits
    
    combined = pd.concat([df_ran_split, df_ben_split], ignore_index=True)
    print("Scenario 1 Ransomware split counts:\n", df_ran_split['Split'].value_counts())
    print("Scenario 1 Total split counts:\n", combined['Split'].value_counts())
    combined.to_csv(SPLIT_SCENARIO1_CSV, index=False)
    print(f"Saved Scenario 1 to {SPLIT_SCENARIO1_CSV}")
    return combined

def generate_scenario2(df_ran, df_ben_split, seed=RANDOM_SEED):
    """
    Scenario 2: Zero-Day / Disjoint Family Split.
    Approximately 70% Train, 15% Val, 15% Test of total malware samples.
    NO family is split across subsets.
    """
    print("\n--- Generating Scenario 2 (Zero-Day Disjoint Split) ---")
    family_counts = df_ran['Family'].value_counts()
    N = len(df_ran)
    target_train = TRAIN_RATIO * N
    target_val = VAL_RATIO * N
    target_test = TEST_RATIO * N
    
    families = family_counts.index.tolist()
    counts = family_counts.values.tolist()
    
    rng = np.random.default_rng(seed)
    best_diff = float('inf')
    best_assignment = None
    
    for iteration in range(2000):
        if iteration == 0:
            indexed_counts = sorted(zip(families, counts), key=lambda x: x[1], reverse=True)
        else:
            top2 = [(families[0], counts[0]), (families[1], counts[1])]
            rest = list(zip(families[2:], counts[2:]))
            rng.shuffle(rest)
            rest.sort(key=lambda x: x[1] * rng.uniform(0.9, 1.1), reverse=True)
            indexed_counts = top2 + rest
            
        train_f, val_f, test_f = [], [], []
        train_c, val_c, test_c = 0, 0, 0
        
        for f, c in indexed_counts:
            if c > target_val:
                train_f.append(f)
                train_c += c
                continue
                
            def_train = (target_train - train_c) / target_train
            def_val = (target_val - val_c) / target_val
            def_test = (target_test - test_c) / target_test
            
            choices = []
            if train_c + c <= target_train + 400:
                choices.append(('train', def_train))
            if val_c + c <= target_val + 400:
                choices.append(('val', def_val))
            if test_c + c <= target_test + 400:
                choices.append(('test', def_test))
                
            if not choices:
                overshoots = [
                    ('train', abs(train_c + c - target_train)),
                    ('val', abs(val_c + c - target_val)),
                    ('test', abs(test_c + c - target_test))
                ]
                best_choice = min(overshoots, key=lambda x: x[1])[0]
            else:
                best_choice = max(choices, key=lambda x: x[1])[0]
                
            if best_choice == 'train':
                train_f.append(f)
                train_c += c
            elif best_choice == 'val':
                val_f.append(f)
                val_c += c
            else:
                test_f.append(f)
                test_c += c
                
        diff = abs(train_c - target_train) + abs(val_c - target_val) + abs(test_c - target_test)
        if diff < best_diff:
            best_diff = diff
            best_assignment = (set(train_f), set(val_f), set(test_f), train_c, val_c, test_c)
            
    train_set, val_set, test_set, tr_c, va_c, te_c = best_assignment
    print(f"Optimal Partition -> Train: {tr_c} ({tr_c/N*100:.2f}%, {len(train_set)} families), "
          f"Val: {va_c} ({va_c/N*100:.2f}%, {len(val_set)} families), "
          f"Test: {te_c} ({te_c/N*100:.2f}%, {len(test_set)} families)")
          
    # Verify disjointness
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0
    print("Zero-Day Disjointness Verified: 100% DISJOINT!")
    
    family_to_split = {}
    for f in train_set: family_to_split[f] = 'train'
    for f in val_set: family_to_split[f] = 'val'
    for f in test_set: family_to_split[f] = 'test'
    
    df_ran_split = df_ran.copy()
    df_ran_split['Split'] = df_ran_split['Family'].map(family_to_split)
    
    combined = pd.concat([df_ran_split, df_ben_split], ignore_index=True)
    combined.to_csv(SPLIT_SCENARIO2_CSV, index=False)
    print(f"Saved Scenario 2 to {SPLIT_SCENARIO2_CSV}")
    return combined

if __name__ == '__main__':
    df_ran, df_ben = load_and_clean_metadata()
    df_ben_split = split_benign(df_ben)
    generate_scenario1(df_ran, df_ben_split)
    generate_scenario2(df_ran, df_ben_split)
    print("\nAll splits generated and verified successfully!")
