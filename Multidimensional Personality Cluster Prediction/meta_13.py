# ============================================================
# SUPERSTACK 13.0 — PART 3
# Meta Model (ElasticNet + Tiny XGB) + 15-Seed Averaging
# ============================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Load meta-features and labels
# ------------------------------------------------------------
S_train = np.load("S_train.npy")   # shape (1913, M*5)
S_test  = np.load("S_test.npy")    # shape (479, M*5)
y_int   = np.load("y_int.npy")

NUM_CLASSES = 5

# ------------------------------------------------------------
# Meta Model A — ElasticNet Logistic Regression
# ------------------------------------------------------------
meta_en = LogisticRegression(
    max_iter=6000,
    solver="saga",
    penalty="elasticnet",
    l1_ratio=0.35,
    class_weight="balanced",
    random_state=42
)

meta_en.fit(S_train, y_int)
enet_oof = meta_en.predict_proba(S_train)
enet_oof_f1 = f1_score(y_int, enet_oof.argmax(axis=1), average="macro")

print(f"\nElasticNet Meta OOF F1 = {enet_oof_f1:.4f}")

# ------------------------------------------------------------
# Meta Model B — Extremely Small XGBoost (acts like a blender)
# ------------------------------------------------------------
meta_xgb = xgb.XGBClassifier(
    n_estimators=180,
    learning_rate=0.06,
    max_depth=3,
    subsample=0.95,
    colsample_bytree=0.95,
    random_state=42,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist"
)

meta_xgb.fit(S_train, y_int)
xgb_oof = meta_xgb.predict_proba(S_train)
xgb_oof_f1 = f1_score(y_int, xgb_oof.argmax(axis=1), average="macro")

print(f"XGB Meta OOF F1 = {xgb_oof_f1:.4f}")

# ------------------------------------------------------------
# Blended Meta Output
# (Weights tuned for highest Kaggle performance)
# ------------------------------------------------------------
blend_oof = 0.42 * enet_oof + 0.58 * xgb_oof
blend_score = f1_score(y_int, blend_oof.argmax(axis=1), average="macro")

print(f"\nBlended Meta OOF F1 = {blend_score:.4f}")

# ------------------------------------------------------------
# Final Prediction — 15 Seed Averaging of XGB Meta
# ------------------------------------------------------------
SEEDS = [42, 101, 2024, 777, 888, 909, 2525, 5050,
         303, 404, 505, 606, 707, 808, 999]

print("\n===================================")
print(" Running 15-Seed XGB Meta Averaging")
print("===================================\n")

final_pred = np.zeros((len(S_test), NUM_CLASSES))

def run_seed(seed):
    model = xgb.XGBClassifier(
        n_estimators=180,
        learning_rate=0.06,
        max_depth=3,
        subsample=0.95,
        colsample_bytree=0.95,
        random_state=seed,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist"
    )
    model.fit(S_train, y_int)
    return model.predict_proba(S_test)

for s in SEEDS:
    print(f" → Seed {s}")
    final_pred += run_seed(s)

final_pred /= len(SEEDS)

# ------------------------------------------------------------
# Map back to class labels
# ------------------------------------------------------------
test = pd.read_csv("test.csv")
classes = ["Cluster_A", "Cluster_B", "Cluster_C", "Cluster_D", "Cluster_E"]

final_labels = final_pred.argmax(axis=1)
final_classes = [classes[i] for i in final_labels]

# ------------------------------------------------------------
# Save submission
# ------------------------------------------------------------
submission = pd.DataFrame({
    "participant_id": test["participant_id"],
    "personality_cluster": final_classes
})

outfile = "submission_superstack_13_5.csv"
submission.to_csv(outfile, index=False)

print("\n====================================================")
print(" FINAL SUBMISSION SAVED:", outfile)
print("====================================================\n")
