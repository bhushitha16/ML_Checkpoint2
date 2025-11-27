import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

# ===========================================
# Columns
# ===========================================

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

print("Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = train[target_col].map({"Stayed": 1, "Left": 0})
X_raw = train[numerical_cols + categorical_cols]
X_test_raw = test[numerical_cols + categorical_cols]

# ===========================================
# Preprocessing pipeline (OneHot + Imputer)
# ===========================================

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

# ===========================================
# Base Models (Simple → Low-Variance)
# ===========================================

base_models = {
    "logreg": LogisticRegression(max_iter=500, C=0.5),
    "ridge": RidgeClassifier(alpha=1.0),
    "svm": LinearSVC(C=0.5),
    "gnb": GaussianNB(),
    "knn": KNeighborsClassifier(n_neighbors=21)
}

# ===========================================
# KFold Setup
# ===========================================

kf = KFold(n_splits=5, shuffle=True, random_state=42)

stack_train = np.zeros((X.shape[0], len(base_models)))   # Meta-features for train
stack_test = np.zeros((X_test.shape[0], len(base_models)))  # Blended test predictions

print("\nTraining base models with 5-Fold CV...\n")

# ===========================================
# Train each base model with CV
# ===========================================

for col_idx, (name, model) in enumerate(base_models.items()):
    print(f"===== {name} =====")

    fold_test_preds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold+1}/5")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        # SVM special: no predict_proba
        if name == "svm":
            val_pred = model.decision_function(X_val)
            test_pred = model.decision_function(X_test)
        else:
            if hasattr(model, "predict_proba"):
                val_pred = model.predict_proba(X_val)[:, 1]
                test_pred = model.predict_proba(X_test)[:, 1]
            else:
                val_pred = model.decision_function(X_val)
                test_pred = model.decision_function(X_test)

        # Store out-of-fold predictions
        stack_train[val_idx, col_idx] = val_pred
        fold_test_preds.append(test_pred)

    # Average test preds across folds
    stack_test[:, col_idx] = np.mean(fold_test_preds, axis=0)
    print()

# ===========================================
# Meta Model (Super Learner)
# ===========================================

print("\nTraining Meta Model (Logistic Regression)...")

meta_model = LogisticRegression(C=0.5, max_iter=500)
meta_model.fit(stack_train, y)

print("Predicting final outputs...")
final_probs = meta_model.predict_proba(stack_test)[:, 1]
final_preds = ["Stayed" if p >= 0.5 else "Left" for p in final_probs]

# ===========================================
# Save submission
# ===========================================

submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: final_preds
})

submission.to_csv("submission_stacking.csv", index=False)
print("\nSaved: submission_stacking.csv")
