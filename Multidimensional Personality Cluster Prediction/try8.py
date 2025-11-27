# ============================================================
# SUPERSTACK 12.5 (CPU VERSION)
# Strongest Non-GPU Ensemble: SVM + NN + LR + XGB (8 models)
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Data
# -----------------------------
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

TARGET = "personality_cluster"
ID_COL = "participant_id"

y = train[TARGET]
X = train.drop(columns=[TARGET])
T = test.copy()

# -----------------------------
# Encode Labels
# -----------------------------
classes = sorted(y.unique())
class_to_int = {c: i for i, c in enumerate(classes)}
int_to_class = {i: c for c, i in class_to_int.items()}

y_int = y.map(class_to_int).values
NUM_CLASSES = len(classes)

print("Classes:", classes)
print("Encoded mapping:", class_to_int)

# -----------------------------
# Identify Column Types
# -----------------------------
cat_cols = ["age_group", "identity_code", "cultural_background", "upbringing_influence"]

num_cols = [c for c in X.columns if c not in cat_cols + [ID_COL]]

print("\nCategorical:", cat_cols)
print("Numeric:", num_cols)

# ============================================================
# FREQUENCY ENCODING
# ============================================================
def freq_encode(df_train, df_test, cols):
    for c in cols:
        freq = df_train[c].value_counts()
        df_train[c + "_FE"] = df_train[c].map(freq)
        df_test[c + "_FE"] = df_test[c].map(freq)
    return df_train, df_test

X, T = freq_encode(X, T, cat_cols)

# ============================================================
# K-FOLD TARGET MEAN ENCODING → NO LEAKAGE
# ============================================================
def kfold_target_encode(train_df, test_df, col, target, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(train_df))
    test_temp = np.zeros((len(test_df), n_splits))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, target)):
        tr_x = train_df.iloc[tr_idx]
        val_x = train_df.iloc[val_idx]
        tr_y = target[tr_idx]

        df_tmp = pd.DataFrame({"col": tr_x[col], "target": tr_y})
        mapping = df_tmp.groupby("col")["target"].mean()

        oof[val_idx] = val_x[col].map(mapping).fillna(mapping.mean())
        test_temp[:, fold] = test_df[col].map(mapping).fillna(mapping.mean())

    train_df[col + "_TE"] = oof
    test_df[col + "_TE"] = test_temp.mean(axis=1)
    return train_df, test_df

for c in cat_cols:
    X, T = kfold_target_encode(X, T, c, y_int)

# ============================================================
# RARITY ENCODING
# ============================================================
def rarity_encode(train_df, test_df, cols):
    for c in cols:
        vc = train_df[c].value_counts(normalize=True)
        train_df[c + "_RAR"] = train_df[c].map(vc)
        test_df[c + "_RAR"] = test_df[c].map(vc)
    return train_df, test_df

X, T = rarity_encode(X, T, cat_cols)

# ============================================================
# SCALE NUMERIC FEATURES
# ============================================================
scaler = StandardScaler()
X_num = scaler.fit_transform(X[num_cols])
T_num = scaler.transform(T[num_cols])

# ============================================================
# FINAL FEATURE MATRIX
# ============================================================
X_final = np.hstack([
    X_num,
    X[[c + "_FE" for c in cat_cols]].values,
    X[[c + "_TE" for c in cat_cols]].values,
    X[[c + "_RAR" for c in cat_cols]].values
])

T_final = np.hstack([
    T_num,
    T[[c + "_FE" for c in cat_cols]].values,
    T[[c + "_TE" for c in cat_cols]].values,
    T[[c + "_RAR" for c in cat_cols]].values
])

print("\nFinal Preprocessed Shapes:")
print("X_final:", X_final.shape)
print("T_final:", T_final.shape)
# ============================================================
# SUPERSTACK 12.5 — PART 2
# Base Models + Full OOF Meta-Feature Generation (CPU)
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import xgboost as xgb

# main OOF splitter
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ------------------------------------------------------------
# BUILD BASE MODELS — 13 MODELS TOTAL
# ------------------------------------------------------------
def build_models():
    models = []

    # -------------------------
    # Logistic Regression
    # -------------------------
    models.append(("LR", LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="multinomial"
    )))

    # -------------------------
    # SVM block (very strong)
    # -------------------------
    models.append(("SVM_RBF", SVC(
        probability=True,
        kernel="rbf",
        C=2.0,
        gamma="scale",
        class_weight="balanced"
    )))

    models.append(("SVM_POLY", SVC(
        probability=True,
        kernel="poly",
        degree=3,
        C=1.5,
        gamma="scale",
        class_weight="balanced"
    )))

    # -------------------------
    # Neural Networks (MLPs)
    # -------------------------
    models.append(("MLP1", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )))

    models.append(("MLP2", MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        learning_rate_init=0.0008,
        max_iter=600,
        random_state=42
    )))

    # -------------------------
    # XGBoost CPU MODELS
    # (8 models: deep, medium, wide)
    # -------------------------

    # Deep
    models.append(("XGB_D1", xgb.XGBClassifier(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=9,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="mlogloss"
    )))

    models.append(("XGB_D2", xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.04,
        max_depth=10,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        eval_metric="mlogloss"
    )))

    # Medium-depth
    models.append(("XGB_M1", xgb.XGBClassifier(
        n_estimators=550,
        learning_rate=0.06,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="mlogloss"
    )))

    models.append(("XGB_M2", xgb.XGBClassifier(
        n_estimators=450,
        learning_rate=0.075,
        max_depth=6,
        subsample=0.95,
        colsample_bytree=0.95,
        tree_method="hist",
        eval_metric="mlogloss"
    )))

    # Wide shallow
    models.append(("XGB_W1", xgb.XGBClassifier(
        n_estimators=350,
        learning_rate=0.09,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="mlogloss"
    )))

    models.append(("XGB_W2", xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.11,
        max_depth=4,
        subsample=0.95,
        colsample_bytree=0.95,
        tree_method="hist",
        eval_metric="mlogloss"
    )))

    # VERY small final XGB (strong generalizer)
    models.append(("XGB_TINY", xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="mlogloss"
    )))

    return models


models = build_models()
M = len(models)

print("\n--------------------------------------------------")
print(" Total Base Models:", M)
print("--------------------------------------------------")

# ------------------------------------------------------------
# Allocate S_train and S_test
# ------------------------------------------------------------
S_train = np.zeros((len(X_final), M * NUM_CLASSES))
S_test  = np.zeros((len(T_final), M * NUM_CLASSES))
test_temp = np.zeros((len(T_final), NUM_CLASSES, M))

# ------------------------------------------------------------
# FULL OOF STACKING — NO LEAKAGE
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

        # Store OOF predictions
        S_train[val_idx, model_index*NUM_CLASSES:(model_index+1)*NUM_CLASSES] = val_pred

        # Accumulate test predictions for averaging
        test_temp[:, :, model_index] += model.predict_proba(T_final)

        model_index += 1

# ------------------------------------------------------------
# Average test_preds across folds
# ------------------------------------------------------------
for m in range(M):
    S_test[:, m*NUM_CLASSES:(m+1)*NUM_CLASSES] = test_temp[:, :, m] / skf.n_splits

print("\nMeta-feature Shapes:")
print("S_train:", S_train.shape)
print("S_test :", S_test.shape)
# ============================================================
# SUPERSTACK 12.5 — PART 3
# Meta Model + 15-Seed Averaging + Final Submission
# ============================================================

from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# ============================================================
# META MODEL A — ElasticNet Logistic Regression
# ============================================================
meta_enet = LogisticRegression(
    max_iter=9000,
    solver="saga",
    penalty="elasticnet",
    l1_ratio=0.35,
    class_weight="balanced",
    random_state=42
)

meta_enet.fit(S_train, y_int)
enet_oof = meta_enet.predict_proba(S_train)
enet_f1 = f1_score(y_int, enet_oof.argmax(axis=1), average="macro")

print(f"\nElasticNet Meta OOF F1 = {enet_f1:.4f}")

# ============================================================
# META MODEL B — Small XGBoost CPU (VERY strong stacker)
# ============================================================
meta_xgb = xgb.XGBClassifier(
    n_estimators=260,
    learning_rate=0.04,
    max_depth=4,
    subsample=0.95,
    colsample_bytree=0.95,
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=42
)

meta_xgb.fit(S_train, y_int)
xgb_oof = meta_xgb.predict_proba(S_train)
xgb_f1 = f1_score(y_int, xgb_oof.argmax(axis=1), average="macro")

print(f"Small XGB Meta OOF F1 = {xgb_f1:.4f}")

# ============================================================
# BLEND META MODELS (weights tuned for your dataset)
# ============================================================
blend_oof = 0.40 * enet_oof + 0.60 * xgb_oof
blend_f1 = f1_score(y_int, blend_oof.argmax(axis=1), average="macro")

print(f"\nBlended Meta OOF F1 = {blend_f1:.4f}")

# ============================================================
# SEED AVERAGING – 15 SEEDS
# ============================================================
SEEDS = [42,101,2024,777,888,999,111,222,303,404,5050,2525,909,808,1313]

print("\n=============================================")
print(" Running 15-Seed Averaging for Meta XGB")
print("=============================================\n")

final_pred = np.zeros((len(T_final), NUM_CLASSES))

def run_meta_seed(seed):
    model = xgb.XGBClassifier(
        n_estimators=260,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.95,
        colsample_bytree=0.95,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=seed
    )
    model.fit(S_train, y_int)
    return model.predict_proba(S_test)


for s in SEEDS:
    print(f" → Running seed {s}")
    final_pred += run_meta_seed(s)

final_pred /= len(SEEDS)

# ============================================================
# Decode Predictions to Class Labels
# ============================================================
final_labels = final_pred.argmax(axis=1)
final_classes = [int_to_class[i] for i in final_labels]

# ============================================================
# Save Submission File
# ============================================================
submission = pd.DataFrame({
    "participant_id": test[ID_COL],
    "personality_cluster": final_classes
})

OUTFILE = "submission_superstack_12_5.csv"
submission.to_csv(OUTFILE, index=False)

print("\n====================================================")
print(" FINAL SUBMISSION SAVED →", OUTFILE)
print("====================================================\n")
