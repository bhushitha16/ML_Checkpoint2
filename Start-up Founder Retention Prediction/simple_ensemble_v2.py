import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

# ======================================
# Column definitions
# ======================================

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

# ======================================
# Load data
# ======================================

print("Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = train[target_col].map({"Stayed": 1, "Left": 0})
X = train[numerical_cols + categorical_cols]
X_test_raw = test[numerical_cols + categorical_cols]

# ======================================
# Preprocessing pipeline
# ======================================

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

# Fit transform
print("Preprocessing...")
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test_raw)

# Split for validation check
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# ======================================
# Strong, robust models only
# ======================================

print("\nTraining models...\n")

models = {}

# Logistic Regression (now class-balanced)
logreg = LogisticRegression(max_iter=600, C=0.3, class_weight="balanced")
models["logreg"] = logreg

# Ridge Classifier with CV
ridge = RidgeClassifierCV(alphas=[0.1, 0.5, 1.0, 2.0], class_weight="balanced")
models["ridge"] = ridge

# Linear SVM + calibration → MUCH better
svm_base = LinearSVC(C=0.5, class_weight="balanced")
svm = CalibratedClassifierCV(svm_base, method="sigmoid", cv=3)
models["svm"] = svm

# Naive Bayes small weight (works surprisingly well on categorical)
gnb = GaussianNB()
models["gnb"] = gnb

val_predictions = {}
test_predictions = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

    # always use calibrated probability
    if hasattr(model, "predict_proba"):
        val_pred = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict_proba(X_test_processed)[:, 1]
    else:
        val_pred = model.decision_function(X_val)
        test_pred = model.decision_function(X_test_processed)

    val_predictions[name] = val_pred
    test_predictions[name] = test_pred

    acc = accuracy_score(y_val, (val_pred >= 0.5).astype(int))
    print(f"{name} Validation Accuracy: {acc:.4f}\n")

# ======================================
# Weighted ensemble (optimized)
# ======================================

weights = {
    "logreg": 0.32,
    "ridge": 0.34,
    "svm": 0.28,
    "gnb": 0.06
}

print("Creating weighted ensemble...")

# Validation score
val_ensemble = np.zeros_like(list(val_predictions.values())[0])
for name in models:
    val_ensemble += weights[name] * val_predictions[name]

val_pred_labels = (val_ensemble >= 0.5).astype(int)
val_acc = accuracy_score(y_val, val_pred_labels)
print(f"\n🔥 Improved Ensemble Validation Accuracy: {val_acc:.4f}")

# ======================================
# Test Predictions
# ======================================

test_ensemble = np.zeros_like(list(test_predictions.values())[0])
for name in models:
    test_ensemble += weights[name] * test_predictions[name]

final_preds = ["Stayed" if p >= 0.5 else "Left" for p in test_ensemble]

submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: final_preds
})

submission.to_csv("submission_simple_ensemble_v2.csv", index=False)
print("\nSaved: submission_simple_ensemble_v2.csv")
