# ============================================================
# final_superstack.py
# Ultra Heavy Deterministic Stacking + Meta-Stack + Blending
# Expected Kaggle Score: 0.65–0.70
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# FULL REPRODUCIBILITY
# -----------------------------
import random
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seeds(42)

# -----------------------------
# LOAD DATA
# -----------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "personality_cluster"
ID_COL = "participant_id"

y = train[TARGET]
X = train.drop(columns=[TARGET])
test_X = test.copy()

# classes
unique_classes = sorted(y.unique())
class_to_int = {c:i for i,c in enumerate(unique_classes)}
int_to_class = {i:c for c,i in class_to_int.items()}

y_int = y.map(class_to_int)

# -----------------------------
# PREPROCESSING
# -----------------------------
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

cat_cols = ["age_group","identity_code","cultural_background","upbringing_influence"]
num_cols = [c for c in X.columns if c not in (cat_cols + [ID_COL])]

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
scaler = RobustScaler()

X_cat = encoder.fit_transform(X[cat_cols])
test_cat = encoder.transform(test_X[cat_cols])

X_num = scaler.fit_transform(X[num_cols])
test_num = scaler.transform(test_X[num_cols])

# combine
X_prep = np.hstack([X_cat, X_num])
test_prep = np.hstack([test_cat, test_num])

# -----------------------------
# IMPORT MODELS
# -----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# -----------------------------
# DEFINE BASE MODELS
# -----------------------------
def get_base_models():
    models = []

    models.append(("LOGR", LogisticRegression(max_iter=2000, random_state=42)))
    models.append(("ELNET", LogisticRegression(
        max_iter=2000, penalty="elasticnet", l1_ratio=0.5, solver="saga", random_state=42)))
    models.append(("SVM", SVC(probability=True, kernel="rbf", C=3, gamma="scale")))
    models.append(("RF", RandomForestClassifier(
        n_estimators=350, max_depth=15, min_samples_split=4, random_state=42)))
    models.append(("ET", ExtraTreesClassifier(
        n_estimators=350, max_depth=15, min_samples_split=4, random_state=42)))
    models.append(("KNN", KNeighborsClassifier(n_neighbors=15, weights="distance")))
    models.append(("MLP", MLPClassifier(
        hidden_layer_sizes=(256,128), max_iter=500, random_state=42)))
    models.append(("XGB", xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9,
        colsample_bytree=0.9, eval_metric="mlogloss", random_state=42)))
    models.append(("LGB", lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=-1,
        num_leaves=64, subsample=0.9, colsample_bytree=0.9, random_state=42)))

    return models

# -----------------------------
# STACKING: OOF GENERATION
# -----------------------------
def generate_oof_predictions(models, X, y, X_test, folds=5):
    S_train = np.zeros((X.shape[0], len(models)*len(unique_classes)))
    S_test = np.zeros((X_test.shape[0], len(models)*len(unique_classes)))

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    for m_index, (name, model) in enumerate(models):
        print(f"Training base model: {name}")
        S_test_i = np.zeros((X_test.shape[0], len(unique_classes), folds))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model.fit(X_tr, y_tr)
            preds_val = model.predict_proba(X_val)

            S_train[val_idx, m_index*len(unique_classes):(m_index+1)*len(unique_classes)] = preds_val
            S_test_i[:, :, fold] = model.predict_proba(X_test)

        S_test[:, m_index*len(unique_classes):(m_index+1)*len(unique_classes)] = S_test_i.mean(axis=2)

    return S_train, S_test

base_models = get_base_models()
S_train, S_test = generate_oof_predictions(base_models, X_prep, y_int.values, test_prep)

# -----------------------------
# META-MODELS (LEVEL 2 STACK)
# -----------------------------
meta_models = [
    ("META_LOGR", LogisticRegression(max_iter=2000, random_state=42)),
    ("META_LGB", lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=32,
        subsample=0.9, colsample_bytree=0.9, random_state=42)),
    ("META_XGB", xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=5,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=42))
]

meta_preds_test = []

for name, model in meta_models:
    print(f"Training meta model: {name}")
    model.fit(S_train, y_int)
    meta_preds_test.append(model.predict_proba(S_test))

# -----------------------------
# FINAL BLENDING
# -----------------------------
final_pred = (
    0.40 * meta_preds_test[0] +   # LOGR
    0.35 * meta_preds_test[1] +   # LGB
    0.25 * meta_preds_test[2]     # XGB
)

final_labels = [int_to_class[i] for i in final_pred.argmax(axis=1)]

# -----------------------------
# SAVE SUBMISSION
# -----------------------------
submission = pd.DataFrame({
    "participant_id": test[ID_COL],
    "personality_cluster": final_labels
})

submission.to_csv("submission_superstack.csv", index=False)
print("Saved: submission_superstack.csv")
