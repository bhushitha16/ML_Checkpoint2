# ============================================================
# ultimate_superstack_v3.py (NO CATBOOST)
# Macro-F1 Optimized Superstack for Kaggle (5 classes)
# Score target: 0.66 – 0.70
# ============================================================

import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.metrics import f1_score

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


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

# Convert classes to integer IDs
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


# ---------- Target Encoding (bug-free version) ----------
def target_encode_kfold(train_df, test_df, col, target, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    global_mean = target.mean()
    encoded = np.zeros(len(train_df))
    test_encoded = np.zeros(len(test_df))

    for tr_idx, val_idx in skf.split(train_df, target):
        tr = train_df.iloc[tr_idx].copy()
        val = train_df.iloc[val_idx].copy()

        # Add target column to tr for groupby
        tr["_y"] = target[tr_idx]

        fold_means = tr.groupby(col)["_y"].mean()

        encoded[val_idx] = val[col].map(fold_means).fillna(global_mean)
        test_encoded += test_df[col].map(fold_means).fillna(global_mean) / n_splits

    train_df[col + "_TE"] = encoded
    test_df[col + "_TE"] = test_encoded

    return train_df, test_df


# Apply target encoding
for c in cat_cols:
    X, test_X = target_encode_kfold(X, test_X, c, y_int)


# ---------- Ordinal Encode ----------
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
encoder.fit(X[cat_cols])

# ---------- Scale ----------
scaler = RobustScaler()
scaler.fit(X[num_cols])


# ---------- Combine All Features ----------
def combine_features(df, df_test):
    feat_list = []
    feat_test_list = []

    feat_list.append(encoder.transform(df[cat_cols]))
    feat_test_list.append(encoder.transform(df_test[cat_cols]))

    fe_cols = [c + "_FE" for c in cat_cols]
    feat_list.append(df[fe_cols].values)
    feat_test_list.append(df_test[fe_cols].values)

    te_cols = [c + "_TE" for c in cat_cols]
    feat_list.append(df[te_cols].values)
    feat_test_list.append(df_test[te_cols].values)

    feat_list.append(scaler.transform(df[num_cols]))
    feat_test_list.append(scaler.transform(df_test[num_cols]))

    return np.hstack(feat_list), np.hstack(feat_test_list)


X_prep, test_prep = combine_features(X, test_X)

print("Final feature shape:", X_prep.shape)


# ============================================================
# BASE MODELS (NO CATBOOST)
# ============================================================
def get_base_models():
    models = []

    models.append(("LOGR", LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    )))

    models.append(("SVM", SVC(
        probability=True, kernel="rbf", C=2.5, gamma="scale",
        class_weight="balanced", random_state=42
    )))

    models.append(("RF", RandomForestClassifier(
        n_estimators=350, max_depth=20, min_samples_split=3,
        class_weight="balanced", random_state=42
    )))

    models.append(("ET", ExtraTreesClassifier(
        n_estimators=350, max_depth=20, min_samples_split=3,
        class_weight="balanced", random_state=42
    )))

    models.append(("KNN", KNeighborsClassifier(
        n_neighbors=17, weights="distance"
    )))

    models.append(("MLP", MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=600,
        random_state=42
    )))

    models.append(("XGB1", xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
        random_state=42
    )))

    models.append(("LGB1", lgb.LGBMClassifier(
        n_estimators=700, learning_rate=0.05, num_leaves=64,
        subsample=0.9, colsample_bytree=0.9,
        objective="multiclass", class_weight="balanced",
        random_state=42
    )))

    return models


# ============================================================
# OOF STACKING (LEVEL 1)
# ============================================================
def generate_oof_predictions(models, X, y, X_test, folds=5):
    S_train = np.zeros((X.shape[0], len(models) * NUM_CLASSES))
    S_test = np.zeros((X_test.shape[0], len(models) * NUM_CLASSES))

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=GLOBAL_SEED)

    for m_idx, (name, model) in enumerate(models):
        print(f"\nTraining base model: {name}")
        S_test_i = np.zeros((X_test.shape[0], NUM_CLASSES, folds))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            model.fit(X_tr, y_tr)
            preds_val = model.predict_proba(X_val)
            preds_test = model.predict_proba(X_test)

            S_train[val_idx, m_idx * NUM_CLASSES:(m_idx + 1) * NUM_CLASSES] = preds_val
            S_test_i[:, :, fold] = preds_test

            f1 = f1_score(y_val, preds_val.argmax(axis=1), average="macro")
            print(f"  Fold {fold+1} F1: {f1:.4f}")

        S_test[:, m_idx * NUM_CLASSES:(m_idx + 1) * NUM_CLASSES] = S_test_i.mean(axis=2)

    return S_train, S_test


base_models = get_base_models()
S_train, S_test = generate_oof_predictions(base_models, X_prep, y_int, test_prep)


# ============================================================
# META MODELS (LEVEL 2)
# ============================================================
meta_models = [
    ("M_LOGR", LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=42
    )),

    ("M_LGB", lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.03,
        num_leaves=48, subsample=0.9, colsample_bytree=0.9,
        objective="multiclass", class_weight="balanced",
        random_state=42
    )),

    ("M_XGB", xgb.XGBClassifier(
        n_estimators=700, learning_rate=0.04,
        max_depth=6, subsample=0.9, colsample_bytree=0.9,
        eval_metric="mlogloss", random_state=42
    ))
]


# Train meta models
meta_preds_test = []
for name, model in meta_models:
    print(f"\nTraining meta model: {name}")
    model.fit(S_train, y_int)
    meta_preds_test.append(model.predict_proba(S_test))


# ============================================================
# BLENDING
# ============================================================
final_pred = (
    0.30 * meta_preds_test[0] +   # LOGR
    0.40 * meta_preds_test[1] +   # LGB
    0.30 * meta_preds_test[2]     # XGB
)

final_labels = [int_to_class[i] for i in final_pred.argmax(axis=1)]


# ============================================================
# SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({
    "participant_id": test[ID_COL],
    "personality_cluster": final_labels
})

submission.to_csv("submission_superstack_v3_no_catboost.csv", index=False)
print("\nSaved: submission_superstack_v3_no_catboost.csv")
