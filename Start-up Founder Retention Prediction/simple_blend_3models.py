import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

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

# ----------------------------
# Strongest Models Only
# ----------------------------

models = {
    "logreg": LogisticRegression(max_iter=500, C=0.6),
    "ridge": RidgeClassifier(alpha=0.8),
    "gnb": GaussianNB()
}

val_preds = {}
test_preds = {}

print("\nTraining 3 Strong Models...\n")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        val_pred = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]
    else:
        df = model.decision_function(X_val)
        df_test = model.decision_function(X_test)
        val_pred = 1/(1 + np.exp(-df))
        test_pred = 1/(1 + np.exp(-df_test))

    val_preds[name] = val_pred
    test_preds[name] = test_pred

    val_acc = accuracy_score(y_val, (val_pred >= 0.5).astype(int))
    print(f"{name} validation accuracy = {val_acc:.4f}\n")

# ----------------------------
# Weighted blending
# ----------------------------

weights = {
    "logreg": 0.45,
    "ridge": 0.35,
    "gnb": 0.20
}

print("Blending predictions...")

final_test_pred = (
    weights["logreg"] * test_preds["logreg"] +
    weights["ridge"] * test_preds["ridge"] +
    weights["gnb"] * test_preds["gnb"]
)

final_label = ["Stayed" if p >= 0.5 else "Left" for p in final_test_pred]

submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: final_label
})

submission.to_csv("submission_simple_blend_3models.csv", index=False)
print("\nSaved: submission_simple_blend_3models.csv")
