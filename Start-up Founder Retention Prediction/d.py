# final_linear_meta_ensemble.py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score

# -----------------------
# Columns from your dataset
# -----------------------
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

# ---------------------------------
# Load Data
# ---------------------------------
print("Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = train[target_col].map({"Stayed": 1, "Left": 0})
X_raw = train[numerical_cols + categorical_cols]
X_test_raw = test[numerical_cols + categorical_cols]

# ---------------------------------
# Preprocessing (IMPORTANT — SAME as your 0.752 code)
# ---------------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols),
])

print("Preprocessing...")
X = preprocessor.fit_transform(X_raw)
X_test = preprocessor.transform(X_test_raw)

print("X shape:", X.shape)

# ---------------------------------
# Train-Val Split
# ---------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ---------------------------------
# 1. RidgeClassifierCV
# ---------------------------------
print("\nTraining RidgeClassifierCV...")
ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))
ridge.fit(X_train, y_train)
ridge_val = ridge.decision_function(X_val)
ridge_test = ridge.decision_function(X_test)

def sigmoid(x):
    x = np.clip(x, -20, 20)
    return 1/(1+np.exp(-x))

ridge_val = sigmoid(ridge_val)
ridge_test = sigmoid(ridge_test)

print("Ridge Val Acc:", accuracy_score(y_val, ridge_val >= 0.5))

# ---------------------------------
# 2. Logistic Regression CV
# ---------------------------------
print("\nTraining LogisticRegressionCV...")
logreg = LogisticRegressionCV(
    Cs=10, cv=3, max_iter=500, n_jobs=-1,
    solver="lbfgs"
)
logreg.fit(X_train, y_train)
logreg_val = logreg.predict_proba(X_val)[:,1]
logreg_test = logreg.predict_proba(X_test)[:,1]

print("LogReg Val Acc:", accuracy_score(y_val, logreg_val >= 0.5))

# ---------------------------------
# 3. Linear SVM (Calibrated)
# ---------------------------------
print("\nTraining LinearSVC + Calibration...")
svc = LinearSVC()
cal_svc = CalibratedClassifierCV(svc, cv=3)
cal_svc.fit(X_train, y_train)

svc_val = cal_svc.predict_proba(X_val)[:,1]
svc_test = cal_svc.predict_proba(X_test)[:,1]

print("SVM Val Acc:", accuracy_score(y_val, svc_val >= 0.5))

# ---------------------------------
# 4. Gaussian Naive Bayes
# ---------------------------------
print("\nTraining GaussianNB...")
gnb = GaussianNB()
gnb.fit(X_train, y_train)

gnb_val = gnb.predict_proba(X_val)[:,1]
gnb_test = gnb.predict_proba(X_test)[:,1]

print("GNB Val Acc:", accuracy_score(y_val, gnb_val >= 0.5))

# ---------------------------------
# 5. SGDClassifier (log-loss) — OPTIONAL but improves meta blend a bit
# ---------------------------------
print("\nTraining SGDClassifier (log-loss)...")
sgd = SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-3)
sgd.fit(X_train, y_train)

sgd_val = sgd.predict_proba(X_val)[:,1]
sgd_test = sgd.predict_proba(X_test)[:,1]

print("SGD Val Acc:", accuracy_score(y_val, sgd_val >= 0.5))

# ---------------------------------
# META ENSEMBLE WEIGHTS
# Tune manually based on validation scores
# ---------------------------------
print("\n--- BLENDING MODELS ---")

# Best performing typically: Ridge + LogReg + SVM 
w_ridge = 0.35
w_logreg = 0.35
w_svm = 0.20
w_gnb = 0.05
w_sgd = 0.05

meta_val = (
    w_ridge * ridge_val +
    w_logreg * logreg_val +
    w_svm * svc_val +
    w_gnb * gnb_val +
    w_sgd * sgd_val
)

meta_test = (
    w_ridge * ridge_test +
    w_logreg * logreg_test +
    w_svm * svc_test +
    w_gnb * gnb_test +
    w_sgd * sgd_test
)

print("META Val Acc:", accuracy_score(y_val, meta_val >= 0.5))

# ---------------------------------
# Save Submission
# ---------------------------------
final_pred = ["Stayed" if p >= 0.5 else "Left" for p in meta_test]

submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: final_pred
})

submission.to_csv("submission_linear_meta.csv", index=False)
print("\nSaved: submission_linear_meta.csv")
