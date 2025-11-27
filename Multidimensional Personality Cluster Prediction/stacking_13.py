# ============================================================
# SUPERSTACK 13.0 — PART 2 (CPU MODE)
# OOF STACKING: LR + SVM + 2×MLP + 8×XGBOOST
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Load preprocessed data from PART 1
# ------------------------------------------------------------
X_final = np.load("X_final.npy")
T_final = np.load("T_final.npy")
y_int   = np.load("y_int.npy")

NUM_CLASSES = 5

# ------------------------------------------------------------
# Base Models
# ------------------------------------------------------------
def build_base_models():
    models = []

    # Logistic Regression
    models.append(("LR", LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="multinomial"
    )))

    # SVM
    models.append(("SVM", SVC(
        probability=True,
        kernel="rbf",
        C=1.8,
        gamma="scale",
        class_weight="balanced"
    )))

    # MLP 1
    models.append(("MLP1", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        learning_rate_init=0.001,
        max_iter=600,
        random_state=42
    )))

    # MLP 2
    models.append(("MLP2", MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        learning_rate_init=0.0008,
        max_iter=700,
        random_state=11
    )))

    # 8× XGBoost (CPU-Optimized Strong Set)
    xgb_params = [
        (650, 9, 0.04),
        (750, 10, 0.03),
        (550, 7, 0.05),
        (450, 6, 0.07),
        (400, 5, 0.08),
        (350, 4, 0.09),
        (300, 6, 0.07),
        (250, 5, 0.10),
    ]

    for i, (n_est, depth, lr) in enumerate(xgb_params):
        models.append((
            f"XGB_{i+1}",
            xgb.XGBClassifier(
                n_estimators=n_est,
                max_depth=depth,
                learning_rate=lr,
                subsample=0.95,
                colsample_bytree=0.95,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=i*10 + 42
            )
        ))

    return models


models = build_base_models()
M = len(models)

print("\n--------------------------------------------------")
print(" Total Base Models:", M)
print("--------------------------------------------------")

# ------------------------------------------------------------
# Allocate Stacking Matrices
# ------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

S_train = np.zeros((len(X_final), M * NUM_CLASSES))
S_test  = np.zeros((len(T_final), M * NUM_CLASSES))
test_temp = np.zeros((len(T_final), NUM_CLASSES, M))

# ------------------------------------------------------------
# OOF Loop
# ------------------------------------------------------------
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_final, y_int)):
    print(f"\n===================== FOLD {fold+1} =====================\n")

    X_tr, X_val = X_final[tr_idx], X_final[val_idx]
    y_tr, y_val = y_int[tr_idx], y_int[val_idx]

    model_index = 0

    for name, model in models:
        print(f"▶ Training {name}")

        model.fit(X_tr, y_tr)
        val_pred = model.predict_proba(X_val)

        fold_f1 = f1_score(y_val, val_pred.argmax(axis=1), average="macro")
        print(f"{name} F1 = {fold_f1:.4f}")

        # store OOF predictions
        S_train[val_idx, model_index*NUM_CLASSES:(model_index+1)*NUM_CLASSES] = val_pred

        # accumulate test predictions
        test_temp[:, :, model_index] += model.predict_proba(T_final)

        model_index += 1


# ------------------------------------------------------------
# Average Test Predictions Across Folds
# ------------------------------------------------------------
for m in range(M):
    S_test[:, m*NUM_CLASSES:(m+1)*NUM_CLASSES] = test_temp[:, :, m] / skf.n_splits

print("\nMeta Feature Shapes:")
print("S_train:", S_train.shape)
print("S_test :", S_test.shape)

# Save for Part 3
np.save("S_train.npy", S_train)
np.save("S_test.npy", S_test)

print("\n✔ Stacking Completed. Proceed to PART 3.")
