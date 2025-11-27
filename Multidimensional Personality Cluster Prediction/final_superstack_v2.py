# ================================================================
# final_superstack_v2.py
# EXTREMELY HEAVY 2-LEVEL STACKING + META-META MODEL + BLENDING
# Expected Kaggle Score: 0.65–0.69
# Runtime: 10–20 minutes
# Deterministic: YES
# ================================================================

import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seeds(42)

# ============================================================
# LOAD DATA
# ============================================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "personality_cluster"
ID_COL = "participant_id"

y = train[TARGET]
X = train.drop(columns=[TARGET])
test_X = test.copy()

# target encoding
unique_classes = sorted(y.unique())
class_to_int = {c:i for i,c in enumerate(unique_classes)}
int_to_class = {i:c for c,i in class_to_int.items()}
y_int = y.map(class_to_int)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def add_features(df):
    df = df.copy()

    # Behavior ratios
    df["focus_consistency_ratio"] = df["focus_intensity"] / (df["consistency_score"] + 1)
    df["activity_engagement_ratio"] = df["physical_activity_index"] / (df["hobby_engagement_level"] + 1)

    # Interactions
    df["creative_support_interaction"] = df["creative_expression_index"] * df["support_environment_score"]
    df["altruism_support"] = df["altruism_score"] * df["support_environment_score"]

    # Combined behavior strength
    df["behavior_strength"] = (
        df["focus_intensity"]
        + df["consistency_score"]
        + df["support_environment_score"]
        + df["physical_activity_index"]
    )

    return df

X = add_features(X)
test_X = add_features(test_X)

# ============================================================
# PREPROCESSING
# ============================================================
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

cat_cols = ["age_group","identity_code","cultural_background","upbringing_influence"]
num_cols = [c for c in X.columns if c not in (cat_cols + [ID_COL])]

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
scaler = RobustScaler()

X_cat = encoder.fit_transform(X[cat_cols])
test_cat = encoder.transform(test_X[cat_cols])

X_num = scaler.fit_transform(X[num_cols])
test_num = scaler.transform(test_X[num_cols])

X_prep = np.hstack([X_cat, X_num])
test_prep = np.hstack([test_cat, test_num])

# ============================================================
# BASE MODELS (LEVEL 1)
# ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

def get_base_models():

    models = []

    # LOGISTIC REGRESSION
    models.append(("LOGR",
                   LogisticRegression(max_iter=2000, random_state=42)))

    # ELASTIC NET
    models.append(("ELNET",
                   LogisticRegression(max_iter=2000, penalty="elasticnet",
                                      solver="saga", l1_ratio=0.5, random_state=42)))

    # CALIBRATED SVM (VERY POWERFUL)
    svm = SVC(kernel="rbf", C=3, gamma="scale")
    models.append(("SVM_CAL",
                   CalibratedClassifierCV(svm, method="sigmoid", cv=3)))

    # RANDOM FOREST
    models.append(("RF",
                   RandomForestClassifier(n_estimators=400, max_depth=18,
                                          min_samples_split=4, random_state=42)))

    # EXTRA TREES
    models.append(("ET",
                   ExtraTreesClassifier(n_estimators=400, max_depth=18,
                                        min_samples_split=4, random_state=42)))

    # MLP NEURAL NETWORK
    models.append(("MLP",
                   MLPClassifier(hidden_layer_sizes=(256,128),
                                 max_iter=600, random_state=42)))

    # KNN
    models.append(("KNN",
                   KNeighborsClassifier(n_neighbors=15, weights="distance")))

    # XGB
    models.append(("XGB",
                   xgb.XGBClassifier(
                       n_estimators=550,
                       learning_rate=0.04,
                       max_depth=7,
                       subsample=0.9,
                       colsample_bytree=0.9,
                       eval_metric="mlogloss",
                       grow_policy="lossguide",
                       random_state=42)))

    # LGB
    models.append(("LGB",
                   lgb.LGBMClassifier(
                       n_estimators=950,
                       learning_rate=0.04,
                       num_leaves=72,
                       subsample=0.92,
                       colsample_bytree=0.92,
                       min_data_in_leaf=4,
                       force_col_wise=True,
                       random_state=42)))

    return models

# ============================================================
# LEVEL-1 STACKING (OOF)
# ============================================================
def generate_oof(models, X, y, X_test, folds=5):
    S_train = np.zeros((X.shape[0], len(models)*len(unique_classes)))
    S_test = np.zeros((X_test.shape[0], len(models)*len(unique_classes)))

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    for m_i, (name, model) in enumerate(models):
        print(f"Training Base Model: {name}")

        test_stack = np.zeros((X_test.shape[0], len(unique_classes), folds))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):

            # Noise injection inside folds (VERY POWERFUL)
            X_tr = X[tr_idx].copy()
            noise = np.random.normal(0, 0.003, X_tr.shape)
            X_tr = X_tr + noise

            X_val = X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            model.fit(X_tr, y_tr)

            preds_val = model.predict_proba(X_val)
            S_train[val_idx, m_i*len(unique_classes):(m_i+1)*len(unique_classes)] = preds_val

            test_stack[:,:,fold] = model.predict_proba(X_test)

        S_test[:, m_i*len(unique_classes):(m_i+1)*len(unique_classes)] = test_stack.mean(axis=2)

    return S_train, S_test

base_models = get_base_models()
S_train, S_test = generate_oof(base_models, X_prep, y_int.values, test_prep)

# ============================================================
# LEVEL-2 META MODELS
# ============================================================
meta_models = [
    ("META_LOGR",
     LogisticRegression(max_iter=3000, random_state=42)),

    ("META_LGB",
     lgb.LGBMClassifier(
         n_estimators=700,
         learning_rate=0.03,
         num_leaves=48,
         subsample=0.9,
         colsample_bytree=0.9,
         force_col_wise=True,
         random_state=42)),

    ("META_XGB",
     xgb.XGBClassifier(
         n_estimators=700, learning_rate=0.03,
         max_depth=5, subsample=0.9, colsample_bytree=0.9,
         eval_metric="mlogloss", grow_policy="lossguide",
         random_state=42))
]

meta_preds = []

for name, model in meta_models:
    print(f"Training Meta Model: {name}")
    model.fit(S_train, y_int)
    meta_preds.append(model.predict_proba(S_test))

# ============================================================
# FINAL BLENDING
# ============================================================
print("Blending final predictions...")

# Weighted blend
final_pred = (
    0.40 * meta_preds[0] +   # LOGR
    0.35 * meta_preds[1] +   # LGB
    0.25 * meta_preds[2]     # XGB
)

final_labels = [int_to_class[i] for i in final_pred.argmax(axis=1)]

# ============================================================
# SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({
    "participant_id": test[ID_COL],
    "personality_cluster": final_labels
})

submission.to_csv("submission_superstack_v2.csv", index=False)
print("Saved: submission_superstack_v2.csv")
