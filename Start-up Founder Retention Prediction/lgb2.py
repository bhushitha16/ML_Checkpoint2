import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ============================================
# Column Definitions
# ============================================
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

# ============================================
# Load Data
# ============================================
print("Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = train[target_col].map({"Stayed": 1, "Left": 0})
X = train[numerical_cols + categorical_cols].copy()
X_test = test[numerical_cols + categorical_cols].copy()

# ============================================
# Preprocessing
# ============================================

# Convert categorical values to strings then category dtype
for col in categorical_cols:
    X[col] = X[col].astype(str).astype("category")
    X_test[col] = X_test[col].astype(str).astype("category")

# Impute numerical missing values
imp = SimpleImputer(strategy="median")
X[numerical_cols] = imp.fit_transform(X[numerical_cols])
X_test[numerical_cols] = imp.transform(X_test[numerical_cols])

# ============================================
# Train-validation split
# ============================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create LightGBM datasets
train_set = lgb.Dataset(
    X_train, label=y_train,
    categorical_feature=categorical_cols
)

val_set = lgb.Dataset(
    X_val, label=y_val,
    categorical_feature=categorical_cols
)

# ============================================
# LightGBM Improved Parameters
# ============================================
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.01,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 4,
    "min_data_in_leaf": 50,
    "lambda_l1": 1,
    "lambda_l2": 1,
    "verbosity": -1,
    "seed": 42
}

# ============================================
# Train Model
# ============================================
print("\nTraining LightGBM Improved Model...\n")

model = lgb.train(
    params,
    train_set,
    num_boost_round=5000,
    valid_sets=[train_set, val_set],
    callbacks=[
        lgb.early_stopping(200),
        lgb.log_evaluation(200)
    ]
)

# ============================================
# Predictions
# ============================================
print("\nPredicting on test set...")
probs = model.predict(X_test)

preds = ["Stayed" if p >= 0.5 else "Left" for p in probs]

submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: preds
})

submission.to_csv("submission_lgb_improved.csv", index=False)
print("\nSaved: submission_lgb_improved.csv")
