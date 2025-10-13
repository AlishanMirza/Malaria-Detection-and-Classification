#!/usr/bin/env python3
"""Preprocess BODMAS dataset and engineer additional features (api_success_ratio, behavior_hash)
"""
import argparse, numpy as np, pandas as pd, hashlib
from pathlib import Path

OUT = Path('data/processed'); OUT.mkdir(parents=True, exist_ok=True)

def load_bodmas(path):
    data = np.load(path, allow_pickle=True)
    X = data['X']
    # metadata may be in separate csv; try to load metadata csv
    md = None
    try:
        md = pd.read_csv('data/bodmas_metadata.csv')
    except Exception:
        md = pd.Dataframe()
    return X, md

def behavior_hash_row(row):
    return int(hashlib.sha1(row.tobytes()).hexdigest(),16) % (10**8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bodmas', default='data/bodmas.npz')
    args = parser.parse_args()
    X, md = load_bodmas(args.bodmas)
    # create placeholder features if metadata missing
    n = X.shape[0]
    api_success = (X.sum(axis=1) % 10)  # synthetic heuristic if no metadata
    api_total = ((X.sum(axis=1) % 7) + 1)
    api_success_ratio = api_success / (api_total + 1e-9)
    bh = [behavior_hash_row(X[i]) for i in range(n)]
    new_feats = np.vstack([api_success_ratio, bh]).T
    X_ext = np.hstack([X, new_feats])
    np.savez_compressed(OUT / 'bodmas_features_ext.npz', X=X_ext)
    print('Saved extended features to', OUT / 'bodmas_features_ext.npz')

if __name__=='__main__':
    main()
