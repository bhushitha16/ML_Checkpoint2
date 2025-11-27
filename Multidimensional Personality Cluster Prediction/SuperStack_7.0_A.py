# ============================================================
# SUPERSTACK 7.0 A — PART 1/3
# Clean Preprocessing + Stable Feature Engineering
# Medium Ensemble (6 XGBoost models)
# Target Macro-F1: 0.63 – 0.65
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def seed_everything(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)

seed_everything(42)

# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

TARGET = "personality_cluster"
ID_COL = "participant_id"

y = train[TARGET]
X = train.drop(columns=[TARGET])
test_X = test.copy()

# ------------------------------------------------------------
# Class Mapping (CORRECT VERSION)
# ------------------------------------------------------------
classes = sorted(y.unique())
class_to_int = {c: i for i, c in enumerate(classes)}
int_to_class = {i: c for c, i in class_to_int.items()}

y_int = y.map(class_to_int).values
NUM_CLASSES = len(classes)

# ------------------------------------------------------------
# Column Groups
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
    "altruism_score",
]

# ------------------------------------------------------------
# Frequency Encoding
# ------------------------------------------------------------
def frequency_encode(train_df, test_df, cols):
    for c in cols:
        freq = train_df[c].value_counts()
        train_df[c + "_FE"] = train_df[c].map(freq)
        test_df[c + "_FE"]  = test_df[c].map(freq)
    return train_df, test_df

X, test_X = frequency_encode(X, test_X, cat_cols)

# ------------------------------------------------------------
# Smooth Target Encoding (Leakage-safe)
# ------------------------------------------------------------
def target_encode(train_df, test_df, col, y, n_splits=5, alpha=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros(len(train_df))
    test_encoded = np.zeros(len(test_df))
    global_mean = y.mean()

    for tr_idx, val_idx in skf.split(train_df, y):
        tr = train_df.iloc[tr_idx][col]
        val = train_df.iloc[val_idx][col]

        tr_y = y[tr_idx]

        df_tr = pd.DataFrame({col: tr, "target": tr_y})

        means = df_tr.groupby(col)["target"].mean()
        counts = df_tr[col].value_counts()

        smooth = ((means * counts) + (global_mean * alpha)) / (counts + alpha)

        oof[val_idx] = val.map(smooth).fillna(global_mean)
        test_encoded += test_df[col].map(smooth).fillna(global_mean) / n_splits

    train_df[col + "_TE"] = oof
    test_df[col + "_TE"] = test_encoded
    return train_df, test_df


for c in cat_cols:
    X, test_X = target_encode(X, test_X, c, y_int)

# ------------------------------------------------------------
# Ordinal Encoding
# ------------------------------------------------------------
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_OE = ord_enc.fit_transform(X[cat_cols])
T_OE = ord_enc.transform(test_X[cat_cols])

# ------------------------------------------------------------
# One-Hot Encoding (low-cardinality only)
# ------------------------------------------------------------
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_OHE = ohe.fit_transform(X[cat_cols])
T_OHE = ohe.transform(test_X[cat_cols])

# ------------------------------------------------------------
# Numeric Scaling
# ------------------------------------------------------------
scaler = RobustScaler()
X_NUM = scaler.fit_transform(X[num_cols])
T_NUM = scaler.transform(test_X[num_cols])

# ------------------------------------------------------------
# Light Interactions (safe-only)
# ------------------------------------------------------------
X["cat12"] = X["age_group"].astype(str) + "_" + X["identity_code"].astype(str)
test_X["cat12"] = test_X["age_group"].astype(str) + "_" + test_X["identity_code"].astype(str)

X["cat34"] = X["cultural_background"].astype(str) + "_" + X["upbringing_influence"].astype(str)
test_X["cat34"] = test_X["cultural_background"].astype(str) + "_" + test_X["upbringing_influence"].astype(str)

inter_cols = ["cat12", "cat34"]

inter_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_INT = inter_enc.fit_transform(X[inter_cols])
T_INT = inter_enc.transform(test_X[inter_cols])

# ------------------------------------------------------------
# Build Final Feature Matrix (balanced, no leakage)
# ------------------------------------------------------------
def build_final_matrix():
    X_final = np.hstack([
        X_OE,
        X_OHE,
        X[[c+"_FE" for c in cat_cols]].values,
        X[[c+"_TE" for c in cat_cols]].values,
        X_NUM,
        X_INT,
    ])

    T_final = np.hstack([
        T_OE,
        T_OHE,
        test_X[[c+"_FE" for c in cat_cols]].values,
        test_X[[c+"_TE" for c in cat_cols]].values,
        T_NUM,
        T_INT,
    ])
    return X_final, T_final

X_final, T_final = build_final_matrix()

print("\nFinal Feature Shapes:")
print("X_final:", X_final.shape)
print("T_final:", T_final.shape)

# ------------------------------------------------------------
# CV Splitter
# ------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# ============================================================
# SUPERSTACK 7.0 A — PART 2/3
# Base Models + 6× XGBoost + Clean OOF Stacking
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb

# ------------------------------------------------------------
# BASE MODELS (stable, no overfit)
# ------------------------------------------------------------
def get_base_models():
    models = []

    # Logistic Regression
    models.append(("LOGR", LogisticRegression(
        max_iter=4000,
        solver="lbfgs",
        multi_class="multinomial",
        class_weight="balanced"
    )))

    # SVM (stable config)
    models.append(("SVM", SVC(
        probability=True,
        kernel="rbf",
        C=2.0,
        gamma="scale",
        class_weight="balanced"
    )))

    # Random Forest
    models.append(("RF", RandomForestClassifier(
        n_estimators=350,
        max_depth=14,
        min_samples_split=3,
        class_weight="balanced",
        random_state=42
    )))

    # Extra Trees
    models.append(("ET", ExtraTreesClassifier(
        n_estimators=350,
        max_depth=14,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42
    )))

    return models


# ------------------------------------------------------------
# XGBOOST ENSEMBLE (6 models)
# ------------------------------------------------------------
def get_xgb_models():
    models = []

    # -------- DEEP MODELS --------
    models.append(("XGB_D1", xgb.XGBClassifier(
        n_estimators=650,
        max_depth=10,
        learning_rate=0.04,
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
        learning_rate=0.045,
        subsample=0.83,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        random_state=2024
    )))

    # -------- MEDIUM MODELS --------
    models.append(("XGB_M1", xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        random_state=101
    )))

    models.append(("XGB_M2", xgb.XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.055,
        subsample=0.92,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        random_state=777
    )))

    # -------- WIDE MODELS --------
    models.append(("XGB_W1", xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.065,
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
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        random_state=2525
    )))

    return models


# ------------------------------------------------------------
# COMBINED LIST
# ------------------------------------------------------------
def get_all_models():
    return get_base_models() + get_xgb_models()


# ------------------------------------------------------------
# CLEAN OOF STACKING PIPELINE
# ------------------------------------------------------------
def run_oof_stacking_all(Xf, y_int, Tf):
    models = get_all_models()
    M = len(models)

    print("\n--------------------------------------------------")
    print(" TOTAL MODELS:", M)
    print("--------------------------------------------------")

    S_train = np.zeros((len(Xf), M * NUM_CLASSES))
    S_test  = np.zeros((len(Tf), M * NUM_CLASSES))

    test_fold_preds = np.zeros((len(Tf), NUM_CLASSES, M))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(Xf, y_int)):
        print(f"\n================ FOLD {fold+1} ================")

        X_tr, X_val = Xf[tr_idx], Xf[val_idx]
        y_tr, y_val = y_int[tr_idx], y_int[val_idx]

        model_index = 0  # RESET for each fold

        for name, model in models:
            print(f"\n▶ Training {name}")

            model.fit(X_tr, y_tr)

            val_pred = model.predict_proba(X_val)
            f1 = f1_score(y_val, val_pred.argmax(axis=1), average="macro")
            print(f"{name} Fold F1 = {f1:.4f}")

            start = model_index * NUM_CLASSES
            end   = (model_index + 1) * NUM_CLASSES

            S_train[val_idx, start:end] = val_pred
            test_fold_preds[:, :, model_index] += model.predict_proba(Tf)

            model_index += 1

    # Average test predictions across folds
    for m in range(M):
        S_test[:, m*NUM_CLASSES:(m+1)*NUM_CLASSES] = test_fold_preds[:, :, m] / skf.get_n_splits()

    return S_train, S_test
# ============================================================
# SUPERSTACK 7.0 A — PART 3/3
# Meta Model + 7-Seed Averaging + Submission
# ============================================================

from sklearn.linear_model import LogisticRegression, RidgeClassifier
import numpy as np

print("\n====================================================")
print(" Running OOF Stacking for Base + XGB Models          ")
print("====================================================\n")

# Generate OOF feature matrices
S_train, S_test = run_oof_stacking_all(X_final, y_int, T_final)

print("\nMeta Feature Shapes:")
print(" S_train:", S_train.shape)
print(" S_test :", S_test.shape)

# ------------------------------------------------------------
# META MODEL (ElasticNet Logistic Regression)
# ------------------------------------------------------------
print("\n====================================================")
print(" TRAINING META MODEL (ElasticNet Logistic Regression)")
print("====================================================\n")

meta_model = LogisticRegression(
    max_iter=6000,
    solver="saga",
    penalty="elasticnet",
    class_weight="balanced",
    l1_ratio=0.4,
    random_state=42
)

meta_model.fit(S_train, y_int)

oof_meta = meta_model.predict_proba(S_train)
oof_f1 = f1_score(y_int, oof_meta.argmax(axis=1), average="macro")

print(f"\nMeta Model OOF Macro-F1 = {oof_f1:.4f}\n")


# ------------------------------------------------------------
# 7-SEED META ENSEMBLE
# ------------------------------------------------------------
SEEDS = [42, 2024, 101, 777, 999, 2525, 5050]

def run_meta_seed(seed):
    model = LogisticRegression(
        max_iter=6000,
        solver="saga",
        penalty="elasticnet",
        class_weight="balanced",
        l1_ratio=0.4,
        random_state=seed
    )
    model.fit(S_train, y_int)
    return model.predict_proba(S_test)

print("===========================================")
print(" Running 7-Seed Meta Averaging")
print("===========================================\n")

final_pred = np.zeros((len(S_test), NUM_CLASSES))

for seed in SEEDS:
    print(f" → Seed {seed}")
    final_pred += run_meta_seed(seed)

final_pred /= len(SEEDS)


# ------------------------------------------------------------
# FINAL LABEL DECODING
# ------------------------------------------------------------
final_labels = final_pred.argmax(axis=1)
final_classes = [int_to_class[int(i)] for i in final_labels]

# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------
submission = pd.DataFrame({
    "participant_id": test[ID_COL],
    "personality_cluster": final_classes
})

output_file = "submission_superstack_7_0_A.csv"
submission.to_csv(output_file, index=False)

print("\n====================================================")
print(" FINAL SUBMISSION SAVED:", output_file)
print("====================================================\n")
