# ==============================================================
# FINAL HEAVY STACKING ENSEMBLE (BEST SCORE VERSION)
# RobustScaler + Numeric Features + Strong Base Models
# Expected Macro F1: 0.62 – 0.70+
# ==============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
import lightgbm as lgb


# ==============================================================
# LOAD DATA
# ==============================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "personality_cluster"

# Encode target
le = LabelEncoder()
y = le.fit_transform(train[TARGET])
X = train.drop(columns=[TARGET])


# ==============================================================
# PREPROCESSING: ROBUST SCALER ON ALL NUMERIC FEATURES
# ==============================================================

scaler = RobustScaler()
scaler.fit(X)

X_scaled = scaler.transform(X)
test_scaled = scaler.transform(test)

# ==============================================================
# K-FOLD STACKING SETUP
# ==============================================================

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

n_classes = len(np.unique(y))
oof_preds = np.zeros((len(X), 9 * n_classes))   # 9 base models


# ==============================================================
# DEFINE BASE MODELS (LEVEL 1)
# ==============================================================

models = [
    ("svm", SVC(C=10, kernel='rbf', gamma='scale',
                class_weight='balanced', probability=True)),

    ("logreg", LogisticRegression(C=0.3, max_iter=500,
                                  class_weight='balanced',
                                  multi_class='multinomial')),

    ("elastic", LogisticRegression(penalty='elasticnet', solver='saga',
                                   l1_ratio=0.5, C=0.2,
                                   max_iter=700, class_weight='balanced')),

    ("rf", RandomForestClassifier(
        n_estimators=400, class_weight='balanced_subsample',
        n_jobs=-1)),

    ("et", ExtraTreesClassifier(
        n_estimators=400, class_weight='balanced', n_jobs=-1)),

    ("knn", KNeighborsClassifier(n_neighbors=23)),

    ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                          activation='relu', alpha=0.0008)),

    ("xgb", XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        eval_metric='mlogloss',
        learning_rate=0.07,
        n_estimators=450,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.2,
        reg_lambda=1.0,
        tree_method='hist',
        random_state=42
    )),

    ("lgb", lgb.LGBMClassifier(
        objective='multiclass',
        num_class=n_classes,
        learning_rate=0.07,
        n_estimators=450,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight='balanced',
        min_child_samples=30,
        reg_alpha=0.2,
        reg_lambda=1.0
    ))
]


# ==============================================================
# TRAIN LEVEL-1 MODELS WITH K-FOLD
# ==============================================================

print("\n🔥 Training Level-1 Models with RobustScaler...\n")

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_scaled, y), 1):
    print(f"\n========== FOLD {fold} ==========")

    X_tr, y_tr = X_scaled[tr_idx], y[tr_idx]
    X_va, y_va = X_scaled[va_idx], y[va_idx]

    col = 0

    for name, model in models:
        print(f" → Training {name.upper()}...")

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)

        oof_preds[va_idx, col:col+n_classes] = proba
        col += n_classes

        f1 = f1_score(y_va, np.argmax(proba, axis=1), average='macro')
        print(f"    {name.upper()} Fold F1: {f1:.4f}")


# ==============================================================
# META MODEL (LEVEL 2)
# ==============================================================

print("\n🔥 Training Meta-Model...\n")

meta_model = LogisticRegression(
    C=0.5,
    max_iter=600,
    multi_class='multinomial'
)

meta_model.fit(oof_preds, y)


# ==============================================================
# TRAIN ALL BASE MODELS ON FULL DATA
# ==============================================================

print("\n🔥 Final Training of Base Models on Full Data...\n")

full_models = []
for name, model in models:
    model.fit(X_scaled, y)
    full_models.append((name, model))


# ==============================================================
# GENERATE META FEATURES FOR TEST DATA
# ==============================================================

print("\n🔥 Generating Test Meta-Features...\n")

test_meta = np.zeros((len(test), 9 * n_classes))
col = 0

for name, model in full_models:
    proba = model.predict_proba(test_scaled)
    test_meta[:, col:col+n_classes] = proba
    col += n_classes

final_pred = meta_model.predict(test_meta)
decoded = le.inverse_transform(final_pred)


# ==============================================================
# SAVE SUBMISSION
# ==============================================================

submission = pd.DataFrame({
    'participant_id': test['participant_id'],
    'personality_cluster': decoded
})

submission.to_csv("submission5.csv", index=False)

print("\n🎉 submission5.csv created successfully!")
print("🔥 This version fixes preprocessing & uses heavy stacking.")
print("🎯 Expect 0.62 – 0.70+ Macro F1 on leaderboard.")
