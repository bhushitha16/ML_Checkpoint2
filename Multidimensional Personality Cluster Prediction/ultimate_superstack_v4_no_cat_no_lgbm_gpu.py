# ============================================================
# ultimate_superstack_v4_no_cat_no_lgbm_gpu.py
# Superstack with GPU XGBoost + SVM + MLP + RF + ET + KNN
# Target: Macro-F1 ≈ 0.66 – 0.69
# ============================================================

import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import xgboost as xgb

# ============================================================
# SEED
# ============================================================
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)

GLOBAL_SEED = 42
set_seeds(GLOBAL_SEED)

# ============================================================
# LOAD DATA
# ============================================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "personality_cluster"
ID_COL = "participant_id"

y = train[TARGET].copy()
X = train.drop(columns=[TARGET])
test_X = test.copy()

unique_classes = sorted(y.unique())
class_to_int = {c: i for i, c in enumerate(unique_classes)}
int_to_class = {i: c for c, i in class_to_int.items()}

y_int = y.map(class_to_int).values
NUM_CLASSES = len(unique_classes)

# ============================================================
# PREPROCESSING
# ============================================================
cat_cols = ["age_group", "identity_code", "cultural_background", "upbringing_influence"]
num_cols = [c for c in X.columns if c not in cat_cols + [ID_COL]]

# ---------- Frequency Encoding ----------
def frequency_encode(train_df, test_df, cols):
    for c in cols:
        freq = train_df[c].value_counts()
        train_df[c + "_FE"] = train_df[c].map(freq)
        test_df[c + "_FE"] = test_df[c].map(freq)
    return train_df, test_df

X, test_X = frequency_encode(X, test_X, cat_cols)

# ---------- SAFE Target Encoding ----------
def target_encode_kfold(train_df, test_df, col, target, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    global_mean = target.mean()
    encoded = np.zeros(len(train_df))
    test_encoded = np.zeros(len(test_df))

    for tr_idx, val_idx in skf.split(train_df, target):
        tr = train_df.iloc[tr_idx].copy()
        val = train_df.iloc[val_idx].copy()

        tr["_y"] = target[tr_idx]
        fold_means = tr.groupby(col)["_y"].mean()

        encoded[val_idx] = val[col].map(fold_means).fillna(global_mean)
        test_encoded += test_df[col].map(fold_means).fillna(global_mean) / n_splits

    train_df[col + "_TE"] = encoded
    test_df[col + "_TE"] = test_encoded

    return train_df, test_df

for c in cat_cols:
    X, test_X = target_encode_kfold(X, test_X, c, y_int)

# ---------- Ordinal Encode ----------
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
encoder.fit(X[cat_cols])

# ---------- Scale ----------
scaler = RobustScaler()
scaler.fit(X[num_cols])

# ---------- Combine Features ----------
def combine_features(df, df_test):

    # Ordinal
    X_cat = encoder.transform(df[cat_cols])
    T_cat = encoder.transform(df_test[cat_cols])

    # FE
    fe_cols = [c + "_FE" for c in cat_cols]
    X_fe = df[fe_cols].values
    T_fe = df_test[fe_cols].values

    # TE
    te_cols = [c + "_TE" for c in cat_cols]
    X_te = df[te_cols].values
    T_te = df_test[te_cols].values

    # Numerics
    X_num = scaler.transform(df[num_cols])
    T_num = scaler.transform(df_test[num_cols])

    X_final = np.hstack([X_cat, X_fe, X_te, X_num])
    T_final = np.hstack([T_cat, T_fe, T_te, T_num])

    return X_final, T_final

X_prep, test_prep = combine_features(X, test_X)
print("Final feature shape:", X_prep.shape)
# ============================================================
# PART 2 — BASE MODELS (NO CATBOOST / NO LGBM) + OOF STACKING
# ============================================================

# ============================================================
# GPU-ENABLED XGBOOST SETTINGS
# ============================================================
gpu_params = {
    "tree_method": "hist",
    "device": "cuda"
}


# ============================================================
# DEFINE BASE MODELS
# ============================================================
def get_base_models():
    models = []

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------
    models.append(("LOGR", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=GLOBAL_SEED
    )))

    # --------------------------------------------------------
    # SVM (with probability)
    # --------------------------------------------------------
    models.append(("SVM", SVC(
        probability=True,
        kernel="rbf",
        C=2.5,
        gamma="scale",
        class_weight="balanced",
        random_state=GLOBAL_SEED
    )))

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------
    models.append(("RF", RandomForestClassifier(
        n_estimators=400,
        max_depth=22,
        min_samples_split=3,
        class_weight="balanced",
        random_state=GLOBAL_SEED
    )))

    # --------------------------------------------------------
    # Extra Trees
    # --------------------------------------------------------
    models.append(("ET", ExtraTreesClassifier(
        n_estimators=450,
        max_depth=22,
        min_samples_split=3,
        class_weight="balanced",
        random_state=GLOBAL_SEED
    )))

    # --------------------------------------------------------
    # Deep Neural Network (MLP)
    # --------------------------------------------------------
    models.append(("MLP", MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        learning_rate_init=0.001,
        max_iter=650,
        random_state=GLOBAL_SEED
    )))

    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------
    models.append(("KNN", KNeighborsClassifier(
        n_neighbors=19,
        weights="distance"
    )))

    # --------------------------------------------------------
    # XGBoost VARIANT 1 — Standard GPU model
    # --------------------------------------------------------
    models.append(("XGB1", xgb.XGBClassifier(
        **gpu_params,
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=GLOBAL_SEED
    )))

    # --------------------------------------------------------
    # XGBoost VARIANT 2 — Deep model
    # --------------------------------------------------------
    models.append(("XGB2", xgb.XGBClassifier(
        **gpu_params,
        n_estimators=800,
        learning_rate=0.03,
        max_depth=9,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=GLOBAL_SEED
    )))

    # --------------------------------------------------------
    # XGBoost VARIANT 3 — Wide model
    # --------------------------------------------------------
    models.append(("XGB3", xgb.XGBClassifier(
        **gpu_params,
        n_estimators=450,
        learning_rate=0.06,
        max_depth=4,
        subsample=0.95,
        colsample_bytree=0.95,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=GLOBAL_SEED
    )))

    # --------------------------------------------------------
    # XGBoost VARIANT 4 — DART boosting
    # --------------------------------------------------------
    models.append(("XGB4", xgb.XGBClassifier(
        **gpu_params,
        booster="dart",
        rate_drop=0.1,
        skip_drop=0.5,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=GLOBAL_SEED
    )))

    return models


# ============================================================
# OOF STACKING FUNCTION
# ============================================================
def generate_oof_predictions(models, X, y, X_test, folds=5):
    S_train = np.zeros((X.shape[0], len(models) * NUM_CLASSES))
    S_test = np.zeros((X_test.shape[0], len(models) * NUM_CLASSES))

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=GLOBAL_SEED)

    for m_idx, (name, model) in enumerate(models):
        print(f"\nTraining base model: {name}")
        S_test_folds = np.zeros((X_test.shape[0], NUM_CLASSES, folds))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            model.fit(X_tr, y_tr)

            val_pred = model.predict_proba(X_val)
            test_pred = model.predict_proba(X_test)

            # store OOF
            start = m_idx * NUM_CLASSES
            end = (m_idx + 1) * NUM_CLASSES
            S_train[val_idx, start:end] = val_pred
            S_test_folds[:, :, fold] = test_pred

            f1 = f1_score(y_val, val_pred.argmax(axis=1), average="macro")
            print(f"  Fold {fold+1} Macro-F1: {f1:.4f}")

        # mean across folds
        S_test[:, start:end] = S_test_folds.mean(axis=2)

    return S_train, S_test


# ============================================================
# RUN LEVEL-1 STACKING
# ============================================================
base_models = get_base_models()

S_train, S_test = generate_oof_predictions(
    base_models, X_prep, y_int, test_prep
)

print("\nOOF Level-1 shape:", S_train.shape)
# ============================================================
# PART 3 — META MODELS + BLENDING + SEED AVERAGING
# ============================================================

# ============================================================
# LEVEL-2 META MODELS
# ============================================================

meta_models = [
    # --------------------------------------------------------
    # Meta Logistic Regression
    # --------------------------------------------------------
    ("META_LOGR", LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        random_state=GLOBAL_SEED
    )),

    # --------------------------------------------------------
    # Meta XGBoost (GPU)
    # --------------------------------------------------------
    ("META_XGB", xgb.XGBClassifier(
        **gpu_params,
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        objective="multi:softprob",
        random_state=GLOBAL_SEED
    )),

    # --------------------------------------------------------
    # ExtraTrees Meta Stabilizer (helps reduce variance)
    # --------------------------------------------------------
    ("META_ET", ExtraTreesClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=3,
        class_weight="balanced",
        random_state=GLOBAL_SEED
    ))
]

meta_preds_test_list = []
meta_preds_train_list = []

print("\n==============================")
print(" TRAINING LEVEL-2 META MODELS ")
print("==============================\n")

for name, model in meta_models:
    print(f"Training meta model: {name}")

    model.fit(S_train, y_int)

    train_pred = model.predict_proba(S_train)
    test_pred = model.predict_proba(S_test)

    meta_preds_train_list.append(train_pred)
    meta_preds_test_list.append(test_pred)

    f1 = f1_score(y_int, train_pred.argmax(axis=1), average="macro")
    print(f"  {name} OOF Macro-F1: {f1:.4f}\n")


# ============================================================
# BLENDING META MODELS
# ============================================================
# Best weights found by testing on L2 validation
blend_pred = (
    0.30 * meta_preds_test_list[0] +     # Logistic Regression
    0.50 * meta_preds_test_list[1] +     # XGBoost (GPU)
    0.20 * meta_preds_test_list[2]       # ExtraTrees
)

blend_labels = blend_pred.argmax(axis=1)
blend_labels = [int_to_class[i] for i in blend_labels]


# ============================================================
# SEED AVERAGING (FINAL BOOST)
# ============================================================

def run_meta_with_seed(seed):
    set_seeds(seed)

    out_preds = []

    for name, model in meta_models:
        # re-seed model if possible
        try:
            model.random_state = seed
        except:
            pass

        model.fit(S_train, y_int)
        out_preds.append(model.predict_proba(S_test))

    # blend again
    return (
        0.30 * out_preds[0] +
        0.50 * out_preds[1] +
        0.20 * out_preds[2]
    )


SEEDS = [42, 2024, 5050]

print("\nRunning seed averaging...")
avg_pred = np.zeros_like(blend_pred)

for s in SEEDS:
    print(f"  Executing seed {s}...")
    avg_pred += run_meta_with_seed(s)

avg_pred /= len(SEEDS)

final_labels = avg_pred.argmax(axis=1)
final_labels = [int_to_class[i] for i in final_labels]


# ============================================================
# SAVE SUBMISSION
# ============================================================

submission = pd.DataFrame({
    "participant_id": test[ID_COL],
    "personality_cluster": final_labels
})

output_file = "submission_superstack_v4_GPU_no_cat_no_lgbm.csv"
submission.to_csv(output_file, index=False)

print("\n======================================================")
print(" FINAL SUBMISSION SAVED:", output_file)
print("======================================================")
