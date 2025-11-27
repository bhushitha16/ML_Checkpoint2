# ============================================================
# SUPERSTACK 8.1 — PART 1/3
# Optimized for SVM (RBF) + Heavy MLP + XGBoost
# Best possible non-CatBoost, non-LGBM stack
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# ------------------------------------------------------------
# FIXED SEED
# ------------------------------------------------------------
def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
seed_everything(42)

# ------------------------------------------------------------
# LOAD DATA (local files)
# ------------------------------------------------------------
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

TARGET = "personality_cluster"
ID_COL = "participant_id"

y = train[TARGET]
X = train.drop(columns=[TARGET])
test_X = test.copy()

# ------------------------------------------------------------
# CLASS ENCODING (safe)
# ------------------------------------------------------------
classes = sorted(list(y.unique()))
class_to_int = {c: i for i, c in enumerate(classes)}
int_to_class = {i: c for c, i in class_to_int.items()}

y_int = y.map(class_to_int).values
NUM_CLASSES = len(classes)

# ------------------------------------------------------------
# COLUMN GROUPS
# ------------------------------------------------------------
cat_cols = [
    "age_group",
    "identity_code",
    "cultural_background",
    "upbringing_influence"
]

num_cols = [
    "focus_intensity",
    "consistency_score",
    "external_guidance_usage",
    "support_environment_score",
    "hobby_engagement_level",
    "physical_activity_index",
    "creative_expression_index",
    "altruism_score"
]

# ------------------------------------------------------------
# FREQUENCY ENCODING
# ------------------------------------------------------------
def frequency_encode(train_df, test_df, cols):
    for c in cols:
        freq = train_df[c].value_counts()
        train_df[c + "_FE"] = train_df[c].map(freq)
        test_df[c + "_FE"]  = test_df[c].map(freq)
    return train_df, test_df

X, test_X = frequency_encode(X, test_X, cat_cols)

# ------------------------------------------------------------
# TARGET ENCODING (SAFE, NO LEAKAGE)
# ------------------------------------------------------------
def target_encode(train_df, test_df, col, y, n_splits=5, alpha=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros(len(train_df))
    test_encoded = np.zeros(len(test_df))

    global_mean = y.mean()

    for tr_idx, val_idx in skf.split(train_df, y):
        tr_c = train_df.iloc[tr_idx][col]
        val_c = train_df.iloc[val_idx][col]
        tr_y = y[tr_idx]

        df_tr = pd.DataFrame({col: tr_c, "target": tr_y})

        means = df_tr.groupby(col)["target"].mean()
        counts = df_tr[col].value_counts()

        smooth = ((means * counts) + (global_mean * alpha)) / (counts + alpha)

        oof[val_idx] = val_c.map(smooth).fillna(global_mean)
        test_encoded += test_df[col].map(smooth).fillna(global_mean) / n_splits

    train_df[col + "_TE"] = oof
    test_df[col + "_TE"] = test_encoded
    return train_df, test_df

for c in cat_cols:
    X, test_X = target_encode(X, test_X, c, y_int)

# ------------------------------------------------------------
# ORDINAL ENCODING (very important for SVM + MLP)
# ------------------------------------------------------------
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_OE = ord_enc.fit_transform(X[cat_cols])
T_OE = ord_enc.transform(test_X[cat_cols])

# ------------------------------------------------------------
# SCALE NUMERICAL FEATURES
# ------------------------------------------------------------
scaler = RobustScaler()
X_NUM = scaler.fit_transform(X[num_cols])
T_NUM = scaler.transform(test_X[num_cols])

# ------------------------------------------------------------
# BUILD FINAL MATRIX (NO OHE, NO INTERACTIONS)
# BEST FOR SVM + MLP
# ------------------------------------------------------------
X_final = np.hstack([
    X_OE,
    X_NUM,
    X[[c + "_FE" for c in cat_cols]].values,
    X[[c + "_TE" for c in cat_cols]].values
])

T_final = np.hstack([
    T_OE,
    T_NUM,
    test_X[[c + "_FE" for c in cat_cols]].values,
    test_X[[c + "_TE" for c in cat_cols]].values
])

print("\nFinal Shapes:")
print("X_final:", X_final.shape)
print("T_final:", T_final.shape)

# ------------------------------------------------------------
# CROSS-VALIDATION (OOF)
# ------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# ============================================================
# SUPERSTACK 8.1 — PART 2/3
# Base Models: SVM (RBF) + Heavy MLP + LR + 6× XGBoost
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# ------------------------------------------------------------
# Logistic Regression + SVM + MLP
# ------------------------------------------------------------
def get_base_models():
    models = []

    # Logistic Regression
    models.append(("LOGR", LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        multi_class="multinomial",
        class_weight="balanced"
    )))

    # RBF SVM (your chosen mode A)
    models.append(("SVM", SVC(
        probability=True,
        kernel="rbf",
        C=2.3,
        gamma="scale",
        class_weight="balanced"
    )))

    # Heavy MLP (Option 3)
    models.append(("MLP", MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation="relu",
        solver="adam",
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=800,
        random_state=42
    )))

    return models


# ------------------------------------------------------------
# XGBoost MODELS (6 models → best setting)
# ------------------------------------------------------------
def get_xgb_models():
    models = []

    # ---------------- DEEP XGBoost ----------------
    models.append(("XGB_D1", xgb.XGBClassifier(
        n_estimators=650,
        max_depth=10,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        random_state=42
    )))

    models.append(("XGB_D2", xgb.XGBClassifier(
        n_estimators=700,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        random_state=2024
    )))

    # ---------------- MEDIUM XGBoost ----------------
    models.append(("XGB_M1", xgb.XGBClassifier(
        n_estimators=850,
        max_depth=6,
        learning_rate=0.045,
        subsample=0.92,
        colsample_bytree=0.82,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        random_state=101
    )))

    models.append(("XGB_M2", xgb.XGBClassifier(
        n_estimators=750,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.88,
        colsample_bytree=0.88,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        random_state=777
    )))

    # ---------------- WIDE XGBoost ----------------
    models.append(("XGB_W1", xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.95,
        colsample_bytree=0.95,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        random_state=999
    )))

    models.append(("XGB_W2", xgb.XGBClassifier(
        n_estimators=550,
        max_depth=4,
        learning_rate=0.065,
        subsample=0.90,
        colsample_bytree=0.90,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        random_state=2525
    )))

    return models


# ------------------------------------------------------------
# ALL BASE MODELS
# ------------------------------------------------------------
def get_all_models():
    return get_base_models() + get_xgb_models()


# ------------------------------------------------------------
# OOF STACKING PIPELINE (Safe, No Leakage)
# ------------------------------------------------------------
def run_oof_stacking_all(Xf, y_int, Tf):
    models = get_all_models()
    M = len(models)

    print("\n--------------------------------------------------")
    print(" Total Base Models (LR + SVM + MLP + XGB):", M)
    print("--------------------------------------------------")

    S_train = np.zeros((len(Xf), M * NUM_CLASSES))
    S_test  = np.zeros((len(Tf), M * NUM_CLASSES))

    test_fold_preds = np.zeros((len(Tf), NUM_CLASSES, M))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(Xf, y_int)):
        print(f"\n===================== FOLD {fold+1} =====================")

        X_tr, X_val = Xf[tr_idx], Xf[val_idx]
        y_tr, y_val = y_int[tr_idx], y_int[val_idx]

        model_idx = 0

        for name, model in models:
            print(f"\n▶ Training: {name}")

            model.fit(X_tr, y_tr)

            val_pred = model.predict_proba(X_val)
            fold_f1 = f1_score(y_val, val_pred.argmax(axis=1), average="macro")
            print(f"{name} Fold F1 = {fold_f1:.4f}")

            start = model_idx * NUM_CLASSES
            end   = (model_idx + 1) * NUM_CLASSES

            S_train[val_idx, start:end] = val_pred
            test_fold_preds[:, :, model_idx] += model.predict_proba(Tf)

            model_idx += 1

    # Average test predictions across folds
    for m in range(M):
        S_test[:, m*NUM_CLASSES:(m+1)*NUM_CLASSES] = \
            test_fold_preds[:, :, m] / skf.get_n_splits()

    return S_train, S_test
# ============================================================
# SUPERSTACK 8.1 — PART 3/3
# Meta Model = XGBoost + 7-Seed Averaging
# ============================================================

print("\n====================================================")
print(" Running OOF Stacking to Generate Meta Features")
print("====================================================\n")

S_train, S_test = run_oof_stacking_all(X_final, y_int, T_final)

print("\nMeta Feature Shapes:")
print("S_train:", S_train.shape)
print("S_test :", S_test.shape)


# ------------------------------------------------------------
# META MODEL → XGBoost (Best meta for this competition)
# ------------------------------------------------------------
def build_meta_model(seed):
    return xgb.XGBClassifier(
        n_estimators=900,
        max_depth=6,
        learning_rate=0.045,
        subsample=0.90,
        colsample_bytree=0.88,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",      # you can change to "cuda" if GPU works
        random_state=seed
    )


# ------------------------------------------------------------
# TRAIN META MODEL (S_train → y_int)
# ------------------------------------------------------------
meta_model = build_meta_model(42)
meta_model.fit(S_train, y_int)

oof_pred = meta_model.predict_proba(S_train)
oof_f1 = f1_score(y_int, oof_pred.argmax(axis=1), average="macro")

print(f"\nMeta Model OOF Macro-F1 = {oof_f1:.4f}\n")


# ------------------------------------------------------------
# 7-SEED AVERAGING FOR FINAL PREDICTIONS
# ------------------------------------------------------------
SEEDS = [42, 101, 777, 999, 2024, 2525, 5050]

print("===============================================")
print(" Performing 7-Seed Meta Averaging...")
print("===============================================\n")

final_pred = np.zeros((S_test.shape[0], NUM_CLASSES))

for seed in SEEDS:
    print(f" → Running seed {seed}")
    model_s = build_meta_model(seed)
    model_s.fit(S_train, y_int)
    final_pred += model_s.predict_proba(S_test)

final_pred /= len(SEEDS)


# ------------------------------------------------------------
# FINAL LABELS
# ------------------------------------------------------------
final_labels = final_pred.argmax(axis=1)
final_clusters = [int_to_class[int(i)] for i in final_labels]


# ------------------------------------------------------------
# SAVE SUBMISSION
# ------------------------------------------------------------
submission = pd.DataFrame({
    "participant_id": test[ID_COL],
    "personality_cluster": final_clusters
})

output_file = "submission_superstack_8_1.csv"
submission.to_csv(output_file, index=False)

print("\n====================================================")
print(" FINAL SUBMISSION SAVED:", output_file)
print("====================================================\n")

