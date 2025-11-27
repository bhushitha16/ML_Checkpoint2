import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# ============================
# CONFIG
# ============================
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


# ============================
# LOAD DATA
# ============================
print("Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = train[target_col].map({"Stayed": 1, "Left": 0})
X = train[numerical_cols + categorical_cols]


# ============================
# PREPROCESSING
# ============================
print("Preprocessing LightGBM data...")

# Numeric imputation
num_imputer = SimpleImputer(strategy="median")
X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
test[numerical_cols] = num_imputer.transform(test[numerical_cols])

# Label encode categoricals
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    mapping = {c: i for i, c in enumerate(le.classes_)}
    test[col] = test[col].astype(str).map(lambda x: mapping.get(x, len(mapping)))
    encoders[col] = le

# Train-val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Build datasets
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val)


# ============================
# LIGHTGBM MODEL
# ============================
params = {
    "objective": "binary",
    "metric": "binary_error",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "seed": 42
}

print("Training LightGBM...")
model = lgb.train(
    params,
    train_set,
    num_boost_round=1000,
    valid_sets=[train_set, val_set],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(100)
    ]
)


# ============================
# PREDICT TEST DATA
# ============================
print("Generating predictions...")

test_X = test[numerical_cols + categorical_cols]
probs = model.predict(test_X)
preds = ["Stayed" if p >= 0.5 else "Left" for p in probs]

submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: preds
})

submission.to_csv("submission_lgb_only.csv", index=False)
print("Saved: submission_lgb_only.csv")
