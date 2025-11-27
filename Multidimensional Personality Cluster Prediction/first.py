# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# ============================================================
# LOAD DATA (LOCAL FILES)
# ============================================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "personality_cluster"

X = train.drop(columns=[TARGET])
y = train[TARGET]

# ============================================================
# PREPROCESSOR (Works for SVM, NN, XGBoost, LightGBM)
# ============================================================

num_cols = [
    "age_group", "identity_code", "cultural_background",
    "upbringing_influence", "focus_intensity",
    "consistency_score", "external_guidance_usage",
    "support_environment_score", "hobby_engagement_level",
    "physical_activity_index", "creative_expression_index",
    "altruism_score"
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols)
    ],
    remainder='drop'
)

# ============================================================
# TRAIN/VALID SPLIT (Full Dataset)
# ============================================================
X_train_full, X_valid_full, y_train_full, y_valid_full = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# MODEL 1: SVM (Full Dataset)
# ============================================================
svm_clf = Pipeline(steps=[
    ('prep', preprocessor),
    ('clf', SVC(kernel='rbf', C=2, gamma='scale'))
])

svm_clf.fit(X_train_full, y_train_full)
svm_pred_full = svm_clf.predict(X_valid_full)
svm_f1_full = f1_score(y_valid_full, svm_pred_full, average="macro")

print("--------------------------------------------------")
print("SVM Macro F1 (Full Dataset):", svm_f1_full)

# ============================================================
# MODEL 2: Neural Network (Full Dataset)
# ============================================================
nn_clf = Pipeline(steps=[
    ('prep', preprocessor),
    ('clf', MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=400,
        random_state=42
    ))
])

nn_clf.fit(X_train_full, y_train_full)
nn_pred_full = nn_clf.predict(X_valid_full)
nn_f1_full = f1_score(y_valid_full, nn_pred_full, average="macro")

print("NN Macro F1 (Full Dataset):", nn_f1_full)
print("--------------------------------------------------")

# ============================================================
# REDUCED DATASET → 20%
# ============================================================
X_small, _, y_small, _ = train_test_split(
    X, y, test_size=0.8, random_state=42, stratify=y
)

X_train_small, X_valid_small, y_train_small, y_valid_small = train_test_split(
    X_small, y_small, test_size=0.2, random_state=42, stratify=y_small
)

# ============================================================
# SVM on 20% Dataset
# ============================================================
svm_small = Pipeline(steps=[
    ('prep', preprocessor),
    ('clf', SVC(kernel='rbf', C=2, gamma='scale'))
])

svm_small.fit(X_train_small, y_train_small)
svm_pred_small = svm_small.predict(X_valid_small)
svm_f1_small = f1_score(y_valid_small, svm_pred_small, average="macro")

print("SVM Macro F1 (20% Dataset):", svm_f1_small)

# ============================================================
# NN on 20% Dataset
# ============================================================
nn_small = Pipeline(steps=[
    ('prep', preprocessor),
    ('clf', MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=400,
        random_state=42
    ))
])

nn_small.fit(X_train_small, y_train_small)
nn_pred_small = nn_small.predict(X_valid_small)
nn_f1_small = f1_score(y_valid_small, nn_pred_small, average="macro")

print("NN Macro F1 (20% Dataset):", nn_f1_small)
print("--------------------------------------------------")

# ============================================================
# FINAL TRAINING ON FULL DATA (FOR SUBMISSION)
# ============================================================
final_model = Pipeline(steps=[
    ('prep', preprocessor),
    ('clf', SVC(kernel='rbf', C=2, gamma='scale'))
])

final_model.fit(X, y)
test_preds = final_model.predict(test)

# Save submission
submission = pd.DataFrame({
    "participant_id": test["participant_id"],
    "personality_cluster": test_preds
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved!")
print("--------------------------------------------------")
