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

print("Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = train[target_col].map({"Stayed": 1, "Left": 0})
X_raw = train[numerical_cols + categorical_cols]
X_test_raw = test[numerical_cols + categorical_cols]

# ======================================
# Preprocessing
# ======================================

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

# Split for sanity validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================================
# Models
# ======================================

logreg = LogisticRegression(C=0.3, max_iter=400)
ridge = RidgeClassifier(alpha=0.8)
nb = GaussianNB()

models = {
    "logreg": logreg,
    "ridge": ridge,
    "nb": nb
}

print("\nTraining models...")
val_preds = {}
test_preds = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        val_pred = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]
    else:  # Ridge
        val_pred = model.decision_function(X_val)
        test_pred = model.decision_function(X_test)

    # Convert ridge logits to probabilities
    if name == "ridge":
        val_pred = 1 / (1 + np.exp(-val_pred))
        test_pred = 1 / (1 + np.exp(-test_pred))

    val_preds[name] = val_pred
    test_preds[name] = test_pred

    val_acc = accuracy_score(y_val, (val_pred >= 0.5).astype(int))
    print(f"{name} Val Accuracy: {val_acc:.4f}\n")

# ======================================
# Weighted Ensemble
# ======================================

weights = {
    "logreg": 0.45,
    "ridge": 0.35,
    "nb": 0.20
}

print("Creating stable weighted blend...")

test_blend = (weights["logreg"] * test_preds["logreg"] +
              weights["ridge"] * test_preds["ridge"] +
              weights["nb"] * test_preds["nb"])

final_preds = ["Stayed" if p >= 0.5 else "Left" for p in test_blend]

submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: final_preds
})

submission.to_csv("submission_simple_blend_v2.csv", index=False)
print("\nSaved: submission_simple_blend_v2.csv")
