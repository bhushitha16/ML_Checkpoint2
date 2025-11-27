import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# ================================
# 1. LOAD DATA
# ================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Store IDs
test_ids = test["participant_id"]

# 🚨 IMPORTANT: Drop participant_id from train and test
X = train.drop(["personality_cluster", "participant_id"], axis=1)
y = train["personality_cluster"]
test_X = test.drop(["participant_id"], axis=1)

# Encode target
label_map = {c: i for i, c in enumerate(y.unique())}
inv_map = {i: c for c, i in label_map.items()}
y = y.map(label_map)

# ================================
# 2. SCALING
# ================================
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_X)

# ================================
# 3. MODELS
# ================================
def get_xgb():
    return XGBClassifier(
        n_estimators=650,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1,
        reg_lambda=2,
        objective="multi:softprob",
        num_class=5,
        tree_method="hist",
        random_state=42
    )

def get_lgbm():
    return lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multiclass",
        class_weight="balanced",
        num_leaves=64,
        num_class=5,
        random_state=42
    )

# ================================
# 4. BLENDING (5-Fold)
# ================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_blend = np.zeros((len(X), 10))
test_blend = np.zeros((len(test), 10))

for f, (tr, va) in enumerate(skf.split(X_scaled, y)):
    X_tr, X_va = X_scaled[tr], X_scaled[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]

    xgb = get_xgb()
    xgb.fit(X_tr, y_tr)
    oof_xgb = xgb.predict_proba(X_va)
    test_xgb = xgb.predict_proba(test_scaled)

    lgbm = get_lgbm()
    lgbm.fit(X_tr, y_tr)
    oof_lgb = lgbm.predict_proba(X_va)
    test_lgb = lgbm.predict_proba(test_scaled)

    oof_blend[va, :5] = oof_xgb
    oof_blend[va, 5:] = oof_lgb
    test_blend += np.hstack([test_xgb, test_lgb]) / 5

    preds = np.argmax((oof_xgb + oof_lgb) / 2, axis=1)
    f1 = f1_score(y_va, preds, average="macro")
    print(f"Fold {f+1} F1 = {f1:.4f}")

# ================================
# 5. BLENDER (Logistic Regression)
# ================================
blender = LogisticRegression(max_iter=500)
blender.fit(oof_blend, y)

final_probs = blender.predict_proba(test_blend)
final_pred = np.argmax(final_probs, axis=1)
final_labels = [inv_map[i] for i in final_pred]

# ================================
# 6. SAVE SUBMISSION
# ================================
sub = pd.DataFrame({
    "participant_id": test_ids,
    "personality_cluster": final_labels
})

sub.to_csv("submission_blend.csv", index=False)
print("Saved submission_blend.csv")
