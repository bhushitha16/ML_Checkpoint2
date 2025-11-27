# ===============================================================
# FINAL OPTUNA PIPELINE: XGBOOST + LIGHTGBM (100 TRIALS EACH)
# Fixed for Label Encoding Error
# ===============================================================

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier

from xgboost import XGBClassifier
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")

# ===============================================================
# LOAD DATA
# ===============================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "personality_cluster"

# ===============================================================
# LABEL ENCODING FIX
# ===============================================================

le = LabelEncoder()
y = le.fit_transform(train[TARGET])   # Now y is numeric integers 0..N-1
X = train.drop(columns=[TARGET])

num_cols = [
    "age_group", "identity_code", "cultural_background",
    "upbringing_influence", "focus_intensity",
    "consistency_score", "external_guidance_usage",
    "support_environment_score", "hobby_engagement_level",
    "physical_activity_index", "creative_expression_index",
    "altruism_score"
]

preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), num_cols)],
    remainder='drop'
)

# ===============================================================
# TRAIN/VALID SPLIT
# ===============================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2,
    random_state=42,
    stratify=y
)

preprocessor.fit(X_train)
X_train_prep = preprocessor.transform(X_train)
X_valid_prep = preprocessor.transform(X_valid)
test_prep = preprocessor.transform(test)

# ===============================================================
# OPTUNA FOR XGBOOST (100 TRIALS)
# ===============================================================

def xgb_objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 200, 900),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
        "objective": "multi:softprob",
        "num_class": len(np.unique(y)),
        "eval_metric": "mlogloss",
        "tree_method": "hist"
    }

    model = XGBClassifier(**params)
    model.fit(X_train_prep, y_train)

    preds = model.predict(X_valid_prep)
    return f1_score(y_valid, preds, average="macro")

print("🔵 Running Optuna for XGBoost (100 trials)...\n")

xgb_study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=100)

print("Best XGB Params:", xgb_study.best_params)
print("Best XGB Score:", xgb_study.best_value)

# ===============================================================
# BEST XGBOOST MODEL
# ===============================================================

best_xgb = XGBClassifier(
    **xgb_study.best_params,
    objective="multi:softprob",
    num_class=len(np.unique(y)),
    eval_metric="mlogloss",
    tree_method="hist"
)

best_xgb.fit(np.vstack([X_train_prep, X_valid_prep]), np.concatenate([y_train, y_valid]))

# ===============================================================
# OPTUNA FOR LIGHTGBM (100 TRIALS)
# ===============================================================

def lgb_objective(trial):

    params = {
        "num_leaves": trial.suggest_int("num_leaves", 10, 200),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 200, 900),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "objective": "multiclass",
        "num_class": len(np.unique(y)),
        "metric": "multi_logloss"
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_prep, y_train)

    preds = model.predict(X_valid_prep)
    return f1_score(y_valid, preds, average="macro")

print("\n🟢 Running Optuna for LightGBM (100 trials)...\n")

lgb_study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
lgb_study.optimize(lgb_objective, n_trials=100)

print("Best LGB Params:", lgb_study.best_params)
print("Best LGB Score:", lgb_study.best_value)

# ===============================================================
# BEST LIGHTGBM MODEL
# ===============================================================

best_lgb = lgb.LGBMClassifier(
    **lgb_study.best_params,
    objective="multiclass",
    num_class=len(np.unique(y)),
    metric="multi_logloss"
)

best_lgb.fit(np.vstack([X_train_prep, X_valid_prep]), np.concatenate([y_train, y_valid]))

# ===============================================================
# ENSEMBLE
# ===============================================================

ensemble = VotingClassifier(
    estimators=[
        ("xgb", best_xgb),
        ("lgb", best_lgb)
    ],
    voting="soft"
)

ensemble.fit(np.vstack([X_train_prep, X_valid_prep]), np.concatenate([y_train, y_valid]))

# ===============================================================
# FINAL PREDICTIONS + LABEL DECODE
# ===============================================================

test_preds = ensemble.predict(test_prep)
decoded_preds = le.inverse_transform(test_preds)

submission = pd.DataFrame({
    "participant_id": test["participant_id"],
    "personality_cluster": decoded_preds
})

submission.to_csv("submission.csv", index=False)
print("\n🎉 FINAL SUBMISSION CREATED: submission.csv (labels decoded)")
