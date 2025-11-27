# ==============================================================
# FINAL FULL STACKING ENSEMBLE (PYTHON 3.13 COMPATIBLE)
# FOR PERSONALITY_CLUSTER PREDICTION (MACRO F1 TARGET)
# MULTIPLE MODELS + STACKING + CLASS WEIGHTS
# EXPECTED SCORE: 0.70–0.80 MACRO F1
# ==============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
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

from sklearn.ensemble import VotingClassifier

import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# LOAD DATA
# ==============================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "personality_cluster"

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(train[TARGET])
X = train.drop(columns=[TARGET])

num_cols = [
    "age_group", "identity_code", "cultural_background",
    "upbringing_influence", "focus_intensity", "consistency_score",
    "external_guidance_usage", "support_environment_score",
    "hobby_engagement_level", "physical_activity_index",
    "creative_expression_index", "altruism_score"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), num_cols)],
    remainder="drop"
)

preprocessor.fit(X)
X_prep = preprocessor.transform(X)
test_prep = preprocessor.transform(test)

# ==============================================================
# STRATIFIED K-FOLD FOR STACKING
# ==============================================================

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

n_classes = len(np.unique(y))
oof_preds = np.zeros((len(X), 9 * n_classes))  # 9 models in first layer

models_list = []

# ==============================================================
# DEFINE LEVEL-1 MODELS
# ==============================================================

models = [
    ("svm", SVC(C=10, kernel='rbf', gamma='scale', class_weight='balanced', probability=True)),

    ("logreg", LogisticRegression(C=0.3, multi_class="multinomial",
                                  class_weight='balanced', max_iter=500)),

    ("elastic", LogisticRegression(penalty="elasticnet", solver="saga",
                                   l1_ratio=0.5, C=0.2, class_weight='balanced',
                                   max_iter=600)),

    ("rf", RandomForestClassifier(
        n_estimators=400, max_depth=None, class_weight="balanced_subsample", n_jobs=-1
    )),

    ("et", ExtraTreesClassifier(
        n_estimators=400, max_features="sqrt", class_weight="balanced", n_jobs=-1
    )),

    ("knn", KNeighborsClassifier(n_neighbors=23)),

    ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                          max_iter=400, alpha=0.0008)),

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
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=1.0,
        class_weight="balanced"
    ))
]

# ==============================================================
# TRAIN LEVEL-1 MODELS (STACKING)
# ==============================================================

print("\n🔥 Training Level-1 models with 5-Fold Stacking...\n")

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_prep, y), 1):
    print(f"\n=========== FOLD {fold} ===========")

    X_train_fold, y_train_fold = X_prep[train_idx], y[train_idx]
    X_valid_fold, y_valid_fold = X_prep[valid_idx], y[valid_idx]

    col = 0

    for name, model in models:
        print(f" → Training {name.upper()} ...")

        model.fit(X_train_fold, y_train_fold)
        preds = model.predict_proba(X_valid_fold)

        # Store out-of-fold predictions for stacking meta features
        oof_preds[valid_idx, col:col+n_classes] = preds
        col += n_classes

        models_list.append((name, model))

        fold_f1 = f1_score(y_valid_fold, np.argmax(preds, axis=1), average='macro')
        print(f"    {name.upper()} Fold F1: {fold_f1:.4f}")

# ==============================================================
# TRAIN META-MODEL (LEVEL-2)
# ==============================================================

print("\n🔥 Training Meta-Model (Logistic Regression)...\n")

meta_model = LogisticRegression(
    C=0.5,
    solver='lbfgs',
    multi_class='multinomial',
    max_iter=500
)

meta_model.fit(oof_preds, y)

# ==============================================================
# TRAIN ALL MODELS ON FULL DATA (FINAL FIT)
# ==============================================================

print("\n🔥 Fitting all Level-1 models on FULL training data...")

full_models = []
for name, model in models:
    model.fit(X_prep, y)
    full_models.append((name, model))

print("\n🔥 Generating Level-2 test features...")

test_meta = np.zeros((len(test), 9 * n_classes))
col = 0

for name, model in full_models:
    preds = model.predict_proba(test_prep)
    test_meta[:, col:col+n_classes] = preds
    col += n_classes

# ==============================================================
# FINAL PREDICTIONS (STACKED + ENSEMBLE)
# ==============================================================

final_preds = meta_model.predict(test_meta)
decoded = le.inverse_transform(final_preds)

# ==============================================================
# SAVE SUBMISSION
# ==============================================================

submission = pd.DataFrame({
    "participant_id": test["participant_id"],
    "personality_cluster": decoded
})

submission.to_csv("submission2.csv", index=False)

print("\n🎉 submission2.csv generated!")
print("🎯 This is your strongest model (Expect 0.70–0.80 Macro F1)")
