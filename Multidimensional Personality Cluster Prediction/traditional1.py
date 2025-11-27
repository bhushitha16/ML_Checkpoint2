# ============================================================
# SUPERSTACK 6.6 LITE  — PART 1/3
# Clean Preprocessing + Stable Features (NO Leakage)
# Target Macro-F1: 0.64 – 0.67
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# ------------------------------------------------------------
# Set seed
# ------------------------------------------------------------
def seed_everything(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
seed_everything(42)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

TARGET = "personality_cluster"
ID_COL = "participant_id"

y = train[TARGET]
X = train.drop(columns=[TARGET])
test_X = test.copy()

# Class mapping
classes = sorted(y.unique())
class_to_int = {c: i for i, c in enumerate(classes)}
int_to_class = {i: c for c, i in class_to_int.items()}


y_int = y.map(class_to_int).values
NUM_CLASSES = len(classes)

# ------------------------------------------------------------
# Define column types
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
# Smooth Target Encoding (OOF, no leakage)
# ------------------------------------------------------------
def target_encode_smooth(train_df, test_df, col, y, n_splits=5, alpha=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(train_df))
    test_encoded = np.zeros(len(test_df))
    global_mean = y.mean()

    for tr_idx, val_idx in skf.split(train_df, y):
        tr = train_df.iloc[tr_idx].copy()
        val = train_df.iloc[val_idx].copy()

        tr["_y"] = y[tr_idx]
        means = tr.groupby(col)["_y"].mean()
        cnts = tr[col].value_counts()

        smooth = ((means * cnts) + (global_mean * alpha)) / (cnts + alpha)

        oof[val_idx] = val[col].map(smooth).fillna(global_mean)
        test_encoded += test_df[col].map(smooth).fillna(global_mean) / n_splits

    train_df[col + "_TE"] = oof
    test_df[col + "_TE"] = test_encoded
    return train_df, test_df

for c in cat_cols:
    X, test_X = target_encode_smooth(X, test_X, c, y_int)

# ------------------------------------------------------------
# Ordinal Encoding
# ------------------------------------------------------------
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_OE = ord_enc.fit_transform(X[cat_cols])
T_OE = ord_enc.transform(test_X[cat_cols])

# ------------------------------------------------------------
# One-Hot Encoding (low-cardinality)
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
# Light categorical interactions (ONLY 2 to avoid overfit)
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
# Build Final Feature Matrix (NO leakage, stable)
# ------------------------------------------------------------
def build_final():
    X_final = np.hstack([
        X_OE,              # ordinal cats
        X_OHE,             # one-hot cats
        X[[c+"_FE" for c in cat_cols]].values,   # freq enc
        X[[c+"_TE" for c in cat_cols]].values,   # target enc
        X_NUM,             # numerical scaled
        X_INT              # interactions
    ])

    T_final = np.hstack([
        T_OE,
        T_OHE,
        test_X[[c+"_FE" for c in cat_cols]].values,
        test_X[[c+"_TE" for c in cat_cols]].values,
        T_NUM,
        T_INT
    ])

    return X_final, T_final

X_final, T_final = build_final()

print("Final Feature Shapes:")
print("X_final:", X_final.shape)
print("T_final:", T_final.shape)

# ------------------------------------------------------------
# Global CV splitter
# ------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# ============================================================
# SUPERSTACK 6.6 LITE — PART 2/3
# Base Models + XGBoost Trio + Clean OOF Stacking
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb

# ------------------------------------------------------------
# BASE MODELS (only stable ones, remove overfit models)
# ------------------------------------------------------------
def get_base_models():
    models = []

    # Logistic Regression (balanced)
    models.append(("LOGR", LogisticRegression(
        max_iter=4000,
        solver="lbfgs",
        multi_class="multinomial",
        class_weight="balanced"
    )))

    # SVM (balanced)
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
# XGBoost Trio (v3.1.1-safe configs)
# ------------------------------------------------------------
def get_xgb_models():
    models = []

    # XGB 1 — Deep
    models.append(("XGB_DEEP", xgb.XGBClassifier(
        n_estimators=600,
        max_depth=9,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        random_state=42
    )))

    # XGB 2 — Medium
    models.append(("XGB_MED", xgb.XGBClassifier(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        random_state=42
    )))

    # XGB 3 — Wide
    models.append(("XGB_WIDE", xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.075,
        subsample=0.9,
        colsample_bytree=0.95,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        random_state=42
    )))

    return models


# ------------------------------------------------------------
# Combined Model List
# ------------------------------------------------------------
def get_all_models():
    return get_base_models() + get_xgb_models()


# ------------------------------------------------------------
# OOF STACKING (Clean, non-leaky version)
# ------------------------------------------------------------
def run_oof_stacking_all(Xf, y_int, Tf):
    models = get_all_models()
    M = len(models)

    print("\n--------------------------------------------------")
    print(" Total Models Used:", M)
    print("--------------------------------------------------\n")

    S_train = np.zeros((len(Xf), M * NUM_CLASSES))
    S_test  = np.zeros((len(Tf), M * NUM_CLASSES))

    # temp storage per model across folds
    test_fold_preds = np.zeros((len(Tf), NUM_CLASSES, M))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(Xf, y_int)):
        print(f"\n===================== FOLD {fold+1} =====================")

        X_tr, X_val = Xf[tr_idx], Xf[val_idx]
        y_tr, y_val = y_int[tr_idx], y_int[val_idx]

        model_index = 0  # RESET per fold

        for name, model in models:
            print(f"\nTraining: {name}")

            model.fit(X_tr, y_tr)

            # Validation predictions
            val_pred = model.predict_proba(X_val)
            f1 = f1_score(y_val, val_pred.argmax(axis=1), average="macro")
            print(f"{name} Fold F1 = {f1:.4f}")

            # Store OOF
            start = model_index * NUM_CLASSES
            end   = (model_index + 1) * NUM_CLASSES
            S_train[val_idx, start:end] = val_pred

            # Store test preds
            test_fold_preds[:, :, model_index] += model.predict_proba(Tf)

            model_index += 1

    # Average test preds
    for m in range(M):
        S_test[:, m*NUM_CLASSES:(m+1)*NUM_CLASSES] = test_fold_preds[:, :, m] / skf.get_n_splits()

    return S_train, S_test
# ============================================================
# SUPERSTACK 6.6 LITE — PART 3/3
# Meta Model (Ridge) + 7-Seed Averaging + Submission
# ============================================================

from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV

print("\n====================================================")
print(" Running OOF Stacking for All Base Models + XGBoost ")
print("====================================================\n")

# Execute stacking
S_train, S_test = run_oof_stacking_all(X_final, y_int, T_final)

print("\nMeta Feature Shapes:")
print("S_train:", S_train.shape)
print("S_test :", S_test.shape)


# ------------------------------------------------------------
# Meta Model: RidgeClassifierCV (Best stability for stacking)
# ------------------------------------------------------------
print("\n====================================================")
print(" Training Ridge Meta-Model (L2 Stacking Layer)")
print("====================================================\n")

meta_model = RidgeClassifierCV(alphas=(0.1, 1.0, 3.0, 5.0))

meta_model.fit(S_train, y_int)

meta_oof_preds = meta_model.predict(S_train)
meta_oof_f1 = f1_score(y_int, meta_oof_preds, average="macro")

print(f"\nMeta Model OOF Macro-F1 = {meta_oof_f1:.4f}\n")


# ------------------------------------------------------------
# 7-SEED AVERAGING (Meta-Level Only)
# ------------------------------------------------------------
SEEDS = [42, 2024, 101, 777, 999, 2525, 5050]

def run_meta_with_seed(seed):
    model = RidgeClassifier(alpha=1.0, random_state=seed)
    model.fit(S_train, y_int)
    return model.decision_function(S_test)  # raw scores → more stable

print("\n==============================")
print(" Running 7-Seed Meta Averaging")
print("==============================\n")

meta_scores = np.zeros((len(S_test), NUM_CLASSES))

for seed in SEEDS:
    print(f" → Seed {seed} ...")
    meta_scores += run_meta_with_seed(seed)

# Average raw scores
meta_scores /= len(SEEDS)

# Softmax to probabilities
def softmax(z):
    expz = np.exp(z - np.max(z, axis=1, keepdims=True))
    return expz / expz.sum(axis=1, keepdims=True)

final_proba = softmax(meta_scores)
final_labels = final_proba.argmax(axis=1)
final_classes = [int_to_class[i] for i in final_labels]


# ------------------------------------------------------------
# SAVE SUBMISSION FILE
# ------------------------------------------------------------
submission = pd.DataFrame({
    "participant_id": test[ID_COL],
    "personality_cluster": final_classes
})

output_file = "submission_superstack_6_6_lite.csv"
submission.to_csv(output_file, index=False)

print("\n====================================================")
print(" FINAL SUBMISSION SAVED:", output_file)
print("====================================================\n")
