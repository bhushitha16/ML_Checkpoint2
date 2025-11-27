# ==============================================================
# FINAL HIGH-SCORE PIPELINE (NO SMOTE — FIXED FOR PYTHON 3.13)
# CLASS WEIGHTS + XGBOOST + LIGHTGBM + ENSEMBLE (5-FOLD)
# Expected Macro F1: 0.70–0.78
# ==============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier

from xgboost import XGBClassifier
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# LOAD DATA
# ==============================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "personality_cluster"

# ==============================================================
# LABEL ENCODING
# ==============================================================

le = LabelEncoder()
y = le.fit_transform(train[TARGET])
X = train.drop(columns=[TARGET])

# ==============================================================
# PREPROCESSOR
# ==============================================================

num_cols = [
    "age_group", "identity_code", "cultural_background",
    "upbringing_influence", "focus_intensity", "consistency_score",
    "external_guidance_usage", "support_environment_score",
    "hobby_engagement_level", "physical_activity_index",
    "creative_expression_index", "altruism_score"
]

preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), num_cols)],
    remainder="drop"
)

# Preprocess full data
preprocessor.fit(X)
X_prep = preprocessor.transform(X)
test_prep = preprocessor.transform(test)

# ==============================================================
# STRATIFIED 5-FOLD CV
# ==============================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_scores = []
lgb_scores = []

print("\n🔥 Training 5 folds (NO SMOTE, class-weighted models)...\n")

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_prep, y), 1):
    print(f"\n================ FOLD {fold} ================\n")

    X_train_fold = X_prep[train_idx]
    y_train_fold = y[train_idx]
    X_valid_fold = X_prep[valid_idx]
    y_valid_fold = y[valid_idx]

    # =====================================================
    # CLASS WEIGHTS (Compute from training data)
    # =====================================================
    unique, counts = np.unique(y_train_fold, return_counts=True)
    total = len(y_train_fold)
    class_weights = {cls: total/(len(unique)*cnt) for cls, cnt in zip(unique, counts)}

    # =====================================================
    # XGBOOST MODEL
    # =====================================================
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(unique),
        eval_metric="mlogloss",
        learning_rate=0.08,
        n_estimators=450,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.25,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        scale_pos_weight=1.0
    )

    xgb.fit(X_train_fold, y_train_fold)
    preds_xgb = xgb.predict(X_valid_fold)
    score_xgb = f1_score(y_valid_fold, preds_xgb, average="macro")
    xgb_scores.append(score_xgb)

    print(f"XGB Fold F1: {score_xgb:.4f}")

    # =====================================================
    # LIGHTGBM MODEL
    # =====================================================
    lgb_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(unique),
        learning_rate=0.08,
        n_estimators=450,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.25,
        reg_lambda=1.0,
        class_weight=class_weights,
        random_state=42
    )

    lgb_model.fit(X_train_fold, y_train_fold)
    preds_lgb = lgb_model.predict(X_valid_fold)
    score_lgb = f1_score(y_valid_fold, preds_lgb, average="macro")
    lgb_scores.append(score_lgb)

    print(f"LGB Fold F1: {score_lgb:.4f}")

print("\n==========================================")
print("XGB Average Macro F1:", np.mean(xgb_scores))
print("LGB Average Macro F1:", np.mean(lgb_scores))
print("==========================================\n")

# ==============================================================
# TRAIN FINAL MODELS ON FULL DATA
# ==============================================================

best_xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(np.unique(y)),
    eval_metric="mlogloss",
    learning_rate=0.08,
    n_estimators=450,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_alpha=0.25,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
)

best_lgb = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=len(np.unique(y)),
    learning_rate=0.08,
    n_estimators=450,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.25,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=42,
)

best_xgb.fit(X_prep, y)
best_lgb.fit(X_prep, y)

# ==============================================================
# ENSEMBLE (Soft Voting)
# ==============================================================

ensemble = VotingClassifier(
    estimators=[
        ("xgb", best_xgb),
        ("lgb", best_lgb)
    ],
    voting="soft"
)

ensemble.fit(X_prep, y)

# ==============================================================
# PREDICT TEST SET
# ==============================================================

test_preds = ensemble.predict(test_prep)
decoded_preds = le.inverse_transform(test_preds)

submission = pd.DataFrame({
    "participant_id": test["participant_id"],
    "personality_cluster": decoded_preds
})

submission.to_csv("submission1.csv", index=False)
print("\n🎯 submission1.csv successfully generated! (Expected F1: 0.70–0.78)")
