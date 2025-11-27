import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Set index columns (not features)
id_col = 'founder_id'
target_col = 'retention_status'

# Identify columns
numerical_cols = [
    'founder_age', 'years_with_startup', 'monthly_revenue_generated', 'funding_rounds_led',
    'distance_from_investor_hub', 'num_dependents', 'years_since_founding'
]
categorical_cols = [
    'founder_gender', 'founder_role', 'work_life_balance_rating', 'venture_satisfaction',
    'startup_performance_rating', 'working_overtime', 'education_background', 'personal_status',
    'startup_stage', 'team_size_category', 'remote_operations', 'leadership_scope',
    'innovation_support', 'startup_reputation', 'founder_visibility'
]

# Preprocessing: Median for numerical, most frequent for categorical, and OneHotEncoder for categorical
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Prepare features and targets
X = train[numerical_cols + categorical_cols]
y = train[target_col].map({'Stayed': 1, 'Left': 0})
X_test = test[numerical_cols + categorical_cols]

# Split train for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Final ML pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

# Train and evaluate
clf.fit(X_train, y_train)
val_preds = clf.predict(X_val)
print("Validation accuracy:", accuracy_score(y_val, val_preds))

# Predict on test set
test_preds = clf.predict(X_test)
test_preds_labels = ['Stayed' if p == 1 else 'Left' for p in test_preds]

# Build submission file
submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: test_preds_labels
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")
