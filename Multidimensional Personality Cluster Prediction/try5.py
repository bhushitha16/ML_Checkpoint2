import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import xgboost as xgb

# -----------------------------
# Load data
# -----------------------------
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

TARGET = "personality_cluster"
ID_COL = "participant_id"

y = train[TARGET]
X = train.drop(columns=[TARGET])
Xt = test.copy()

# -----------------------------
# Encode classes
# -----------------------------
classes = sorted(y.unique())
c2i = {c:i for i,c in enumerate(classes)}
i2c = {i:c for c,i in c2i.items()}

y_int = y.map(c2i).values
NUM_CLASSES = len(classes)

# -----------------------------
# Categorical + Numeric
# -----------------------------
cat_cols = ["age_group","identity_code","cultural_background","upbringing_influence"]
num_cols = [
    "focus_intensity","consistency_score","external_guidance_usage",
    "support_environment_score","hobby_engagement_level",
    "physical_activity_index","creative_expression_index","altruism_score"
]

# -----------------------------
# Ordinal Encode
# -----------------------------
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_cat = oe.fit_transform(X[cat_cols])
T_cat = oe.transform(Xt[cat_cols])

# -----------------------------
# Robust Scaling
# -----------------------------
scaler = RobustScaler()
X_num = scaler.fit_transform(X[num_cols])
T_num = scaler.transform(Xt[num_cols])

# -----------------------------
# Final features
# -----------------------------
Xf = np.hstack([X_cat, X_num])
Tf = np.hstack([T_cat, T_num])

print("Final shape:", Xf.shape)

# ==================================================
# 1️⃣ TRAIN XGBoost (main model)
# ==================================================
xgb_model = xgb.XGBClassifier(
    n_estimators=900,
    max_depth=6,
    learning_rate=0.045,
    subsample=0.90,
    colsample_bytree=0.88,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
)

xgb_model.fit(Xf, y_int)
pred_xgb = xgb_model.predict_proba(Tf)

# ==================================================
# 2️⃣ TRAIN SVM (helps minority classes)
# ==================================================
svm_model = SVC(
    probability=True,
    kernel="rbf",
    C=2.3,
    gamma="scale",
    class_weight="balanced"
)
svm_model.fit(Xf, y_int)
pred_svm = svm_model.predict_proba(Tf)

# ==================================================
# 3️⃣ TRAIN LOGISTIC REGRESSION (stability)
# ==================================================
lr_model = LogisticRegression(
    max_iter=6000,
    solver="lbfgs",
    multi_class="multinomial",
    class_weight="balanced"
)
lr_model.fit(Xf, y_int)
pred_lr = lr_model.predict_proba(Tf)

# ==================================================
# ⭐ FINAL BLENDING (best weights for your dataset)
# ==================================================
final_pred = (
    0.60 * pred_xgb +
    0.25 * pred_svm +
    0.15 * pred_lr
)

# ==================================================
# Submission
# ==================================================
final_labels = final_pred.argmax(axis=1)
final_clusters = [i2c[i] for i in final_labels]

submission = pd.DataFrame({
    "participant_id": test[ID_COL],
    "personality_cluster": final_clusters
})

submission.to_csv("submission_final_boost_ensemble.csv", index=False)
print("\nSaved: submission_final_boost_ensemble.csv")
