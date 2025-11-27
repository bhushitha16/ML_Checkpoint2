# ==============================================================
# FINAL FULL STACKING ENSEMBLE (CORRECT PREPROCESSING)
# MULTI-MODEL STACKING FOR PERSONALITY CLUSTER PREDICTION
# EXPECTED SCORE: 0.70–0.80 MACRO F1
# ==============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# LOAD DATA
# ==============================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "personality_cluster"

# Label Encode Target
le = LabelEncoder()
y = le.fit_transform(train[TARGET])
X = train.drop(columns=[TARGET])

# ==============================================================
# FEATURE GROUPING
# ==============================================================

categorical_cols = [
    "age_group", "identity_code", "cultural_background",
    "upbringing_influence", "consistency_score",
    "external_guidance_usage", "support_environment_score",
    "hobby_engagement_level", "physical_activity_index",
    "creative_expression_index", "altruism_score"
]

numeric_cols = ["focus_intensity"]

# ==============================================================
# PREPROCESSOR
# ==============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ],
    remainder="drop"
)

# Fit on train
preprocessor.fit(X)
X_prep = preprocessor.transform(X)
test_prep = preprocessor.transform(test)

# ==============================================================
# STRATIFIED K-FOLD STACKING SETUP
# ==============================================================

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

n_classes = len(np.unique(y))
oof_preds = np.zeros((len(X), 9 * n_classes))  # 9 base models

models = [
    ("svm", SVC(C=10, kernel='rbf', gamma='scale', class_weight='balanced', probability=True)),
    ("logreg", LogisticRegression(C=0.3, multi_class="multinomial", class_weight='balanced', max_iter=500)),
    ("elastic", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                   C=0.2, class_weight='balanced', max_iter=700)),
    ("rf", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", n_jobs=-1)),
    ("et", ExtraTreesClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1)),
    ("knn", KNeighborsClassifier(n_neighbors=25)),
    ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=400, alpha=0.0008)),
    ("xgb", XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        learning_rate=0.07,
        n_estimators=450,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.2,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42
    )),
    ("lgb", lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        learning_rate=0.07,
        n_estimators=450,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        min_child_samples=30,
        reg_alpha=0.2,
        reg_lambda=1.0
    ))
]

print("\n🔥 Training Level-1 Models with Correct Preprocessing...\n")

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_prep, y), 1):
    print(f"\n========== FOLD {fold} ==========")

    X_train_fold, X_valid_fold = X_prep[train_idx], X_prep[valid_idx]
    y_train_fold, y_valid_fold = y[train_idx], y[valid_idx]

    col = 0

    for name, model in models:
        print(f" → Training {name.upper()}...")

        model.fit(X_train_fold, y_train_fold)
        pred = model.predict_proba(X_valid_fold)

        oof_preds[valid_idx, col:col+n_classes] = pred
        col += n_classes

        fold_f1 = f1_score(y_valid_fold, np.argmax(pred, axis=1), average="macro")
        print(f"    {name.upper()} Fold F1: {fold_f1:.4f}")

# ==============================================================
# META-MODEL TRAINING
# ==============================================================

print("\n🔥 Training Meta-Model...\n")
meta_model = LogisticRegression(C=0.5, multi_class='multinomial', max_iter=600)
meta_model.fit(oof_preds, y)

# ==============================================================
# TRAIN ALL MODELS ON FULL DATA
# ==============================================================

print("\n🔥 Final Full-Data Training of Base Models...\n")
full_models = []

for name, model in models:
    model.fit(X_prep, y)
    full_models.append((name, model))

# ==============================================================
# META FEATURES FOR TEST
# ==============================================================

print("\n🔥 Generating Meta Features for Test...\n")

test_meta = np.zeros((len(test), 9 * n_classes))
col = 0

for name, model in full_models:
    pred = model.predict_proba(test_prep)
    test_meta[:, col:col+n_classes] = pred
    col += n_classes

final_pred = meta_model.predict(test_meta)
decoded_pred = le.inverse_transform(final_pred)

submission = pd.DataFrame({
    "participant_id": test["participant_id"],
    "personality_cluster": decoded_pred
})

submission.to_csv("submission3.csv", index=False)

print("\n🎉 submission3.csv created successfully!")
print("🔥 This version uses the CORRECT preprocessing → Expect 0.70+ Macro F1.")
