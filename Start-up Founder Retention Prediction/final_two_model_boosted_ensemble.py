import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import xgboost as xgb
import lightgbm as lgb

# =========================
# Column lists
# =========================
numerical_cols = [
    'founder_age', 'years_with_startup', 'monthly_revenue_generated',
    'funding_rounds_led', 'distance_from_investor_hub',
    'num_dependents', 'years_since_founding'
]

categorical_cols = [
    'founder_gender', 'founder_role', 'work_life_balance_rating',
    'venture_satisfaction', 'startup_performance_rating', 'working_overtime',
    'education_background', 'personal_status', 'startup_stage',
    'team_size_category', 'remote_operations', 'leadership_scope',
    'innovation_support', 'startup_reputation', 'founder_visibility'
]

target_col = "retention_status"
id_col = "founder_id"

print("\nLoading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = train[target_col].map({"Stayed": 1, "Left": 0})
X_raw = train[numerical_cols + categorical_cols]
X_test_raw = test[numerical_cols + categorical_cols]

# =========================
# Preprocessing
# =========================
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

print("Preprocessing...")
X = preprocessor.fit_transform(X_raw)
X_test = preprocessor.transform(X_test_raw)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ======================================================
# MODEL 1: XGBoost
# ======================================================
print("\nTraining XGBoost...")

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
}

xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=50
)

xgb_test_pred = xgb_model.predict(dtest)

# ======================================================
# MODEL 2: LightGBM
# ======================================================
print("\nTraining LightGBM...")

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val)

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 4,
    "min_data_in_leaf": 40,
    "lambda_l2": 2.0,
}

lgb_model = lgb.train(
    params=lgb_params,
    train_set=lgb_train,
    valid_sets=[lgb_train, lgb_val],
    num_boost_round=800,
    callbacks=[
        lgb.early_stopping(60),
        lgb.log_evaluation(100)
    ]
)

lgb_test_pred = lgb_model.predict(X_test)

# ======================================================
# FINAL ENSEMBLE (BEST SETUP)
# ======================================================
print("\nCreating final boosted ensemble...")

final_pred = (
    0.60 * lgb_test_pred +
    0.40 * xgb_test_pred
)

final_label = ["Stayed" if p >= 0.5 else "Left" for p in final_pred]

submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: final_label
})

submission.to_csv("submission_two_model_ensemble.csv", index=False)
print("\nSaved: submission_two_model_ensemble.csv")
