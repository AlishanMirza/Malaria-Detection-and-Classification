
# Multi-Class Malware Family Training Script (GPU)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import joblib
import logging
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import optuna

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===================================================
# CONFIG
# ===================================================
DATA_PATH = "BODMAS/bodmas.npz"
META_PATH = "BODMAS/bodmas_metadata.csv"
RANDOM_STATE = 42
OPTUNA_TRIALS = 25
TOP_FEATURES = 200

# ===================================================
# 1. LOAD DATA
# ===================================================
logging.info("Loading BODMAS dataset...")
data = np.load(DATA_PATH)
X_all = data["X"]
y_binary = data["y"]  # original binary labels, will ignore
logging.info(f"Features shape: {X_all.shape}, Binary labels shape: {y_binary.shape}")

meta = pd.read_csv(META_PATH)
logging.info(f"Metadata shape: {meta.shape}")

# Identify family column
family_col_candidates = ["family", "category", "family_name"]
for col in family_col_candidates:
    if col in meta.columns:
        family_col = col
        break
else:
    raise ValueError("No valid family column found in metadata!")

# ===================================================
# 2. FILTER MALWARE (remove benign)
# ===================================================
logging.info(f"Original metadata has {len(meta)} samples")

# Malicious if family column is not empty/NaN
malicious_mask = meta[family_col].notna()
X = X_all[malicious_mask.values]
y_families = meta.loc[malicious_mask, family_col].values

# Remove rare families (<5 samples)
family_counts = pd.Series(y_families).value_counts()
families_to_keep = family_counts[family_counts >= 5].index
keep_mask = pd.Series(y_families).isin(families_to_keep).values
X = X[keep_mask]
y_families = y_families[keep_mask]

logging.info(f"Filtered to {X.shape[0]} malicious samples (>=5 samples per family)")

# Encode family names
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_families)
label_to_family = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
logging.info(f"Training on {len(label_to_family)} families")

# ===================================================
# 3. TRAIN / VALIDATION / TEST SPLIT (70/15/15)
# ===================================================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=RANDOM_STATE, stratify=y_trainval
)  # 0.1765 ≈ 15% of total

logging.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# ===================================================
# 4. SCALE FEATURES
# ===================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ===================================================
# 5. FEATURE SELECTION
# ===================================================
selector = SelectKBest(f_classif, k=min(TOP_FEATURES, X_train.shape[1]))
X_train = selector.fit_transform(X_train, y_train)
X_val = selector.transform(X_val)
X_test = selector.transform(X_test)

# ===================================================
# 6. OPTUNA HYPERPARAMETER TUNING (GPU)
# ===================================================
def optuna_objective(trial):
    param = {
        "objective": "multiclass",
        "num_class": len(np.unique(y_train)),
        "metric": "multi_logloss",
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": RANDOM_STATE,
        "device_type": "gpu",       # use GPU
        "gpu_platform_id": 1,       # NVIDIA platform
        "gpu_device_id": 0,         # RTX 4050
    }

    clf = lgb.LGBMClassifier(**param)
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(50)],
        verbose=False
    )
    preds = clf.predict(X_val)
    return f1_score(y_val, preds, average="macro")


logging.info("Starting Optuna hyperparameter tuning on GPU...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(optuna_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
best_params = study.best_params
logging.info(f"Best parameters: {best_params}")

# ===================================================
# 7. TRAIN FINAL MODEL (Train + Val on GPU)
# ===================================================
final_model = lgb.LGBMClassifier(
    **best_params,
    objective="multiclass",
    num_class=len(np.unique(y_trainval)),
    metric="multi_logloss",
    random_state=RANDOM_STATE,
    device_type="gpu",
    gpu_platform_id=1,
    gpu_device_id=0
)
final_model.fit(np.vstack((X_train, X_val)), np.hstack((y_train, y_val)))

# ===================================================
# 8. EVALUATION (Top-20 Families)
# ===================================================
preds_test = final_model.predict(X_test)
probs_test = final_model.predict_proba(X_test)

acc = accuracy_score(y_test, preds_test)
f1 = f1_score(y_test, preds_test, average="macro")

print("\n=== Final Model Performance ===")
print(f"Overall Accuracy: {acc:.4f}")
print(f"Overall F1 Score (Macro): {f1:.4f}")

# Top 20 families in test set
top_families_indices, _ = zip(
    *sorted(pd.Series(y_test).value_counts().items(), key=lambda x: x[1], reverse=True)[:20]
)
top_families_names = label_encoder.inverse_transform(list(top_families_indices))

print("\nClassification Report (Top 20 Families in Test Set):\n")
print(classification_report(y_test, preds_test, labels=top_families_indices, target_names=top_families_names))

# Confusion matrix for top-20
cm = confusion_matrix(y_test, preds_test, labels=top_families_indices)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=top_families_names, yticklabels=top_families_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Top 20 Most Frequent Families)")
plt.tight_layout()
plt.show()

# ===================================================
# 9. SAVE ARTIFACTS
# ===================================================
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/final_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(selector, "models/selector.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(label_to_family, "models/label_to_family.pkl")
logging.info("Saved model, scaler, selector, label_encoder, and label_to_family")

# ===================================================
# 10. SAVE PREDICTIONS
# ===================================================
pred_family_names = label_encoder.inverse_transform(preds_test)
results_df = pd.DataFrame(preds_test, columns=["label"])
results_df["family"] = pred_family_names
results_df.to_csv("predictions_all_samples.csv", index=False)
logging.info("Saved predictions to predictions_all_samples.csv")