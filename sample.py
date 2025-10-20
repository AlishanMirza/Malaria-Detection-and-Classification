import numpy as np
import pandas as pd
import os

# --- CONFIG ---
DATA_PATH = "BODMAS/bodmas.npz"
META_PATH = "BODMAS/bodmas_metadata.csv"

def get_first_malware_sample():
    """
    Loads the BODMAS dataset, finds the first malicious sample,
    and returns its feature vector and family name.
    """
    if not os.path.exists(DATA_PATH) or not os.path.exists(META_PATH):
        print(f"❌ Error: Could not find dataset files at {DATA_PATH} or {META_PATH}")
        return None, None

    X_all = np.load(DATA_PATH)["X"]
    meta = pd.read_csv(META_PATH)

    family_col = next((col for col in ["family", "category", "family_name"] if col in meta.columns), None)
    if not family_col:
        print("❌ Error: Could not find family column in metadata.")
        return None, None
        
    malicious_mask = meta[family_col].notna()
    if not malicious_mask.any():
        print("❌ Error: No malicious samples found in metadata.")
        return None, None
    first_malicious_index = malicious_mask.idxmax()
    
    sample_features = X_all[first_malicious_index]
    sample_family = meta.loc[first_malicious_index, family_col]

    return list(sample_features), sample_family

def get_n_random_malware_samples(num_samples=10):
    """
    Loads the BODMAS dataset and returns a specified number of random malware samples.
    """
    if not os.path.exists(DATA_PATH) or not os.path.exists(META_PATH):
        return [], []

    X_all = np.load(DATA_PATH)["X"]
    meta = pd.read_csv(META_PATH)
    
    family_col = next((col for col in ["family", "category", "family_name"] if col in meta.columns), None)
    if not family_col:
        return [], []

    malicious_mask = meta[family_col].notna()
    malicious_indices = meta.index[malicious_mask].tolist()

    if len(malicious_indices) < num_samples:
        print(f"⚠️ Warning: Requested {num_samples} samples, but only {len(malicious_indices)} malicious samples are available.")
        num_samples = len(malicious_indices)

    random_indices = np.random.choice(malicious_indices, size=num_samples, replace=False)
    
    features_list = [list(X_all[i]) for i in random_indices]
    families_list = meta.loc[random_indices, family_col].tolist()

    return features_list, families_list

if __name__ == "__main__":
    print("🔎 Finding a real malware sample from your dataset...")
    features, family = get_first_malware_sample()
    
    if features:
        print(f"✅ Found a sample from the '{family}' family.")
        print("\n📋 Feature vector (list of numbers):\n")
        print(features)
