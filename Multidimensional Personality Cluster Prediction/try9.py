# ============================================================
#                       SUPERSTACK 15.0
#        SVM + NN + XGB + LR • NO LEAKAGE • MACRO F1
#          Optimized for 5-Class Personality Dataset
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Load data
# -----------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "personality_cluster"
ID_COL = "participant_id"

y = train[TARGET]
X = train.drop(columns=[TARGET])
T = test.copy()

# Encode target to integers
classes = sorted(y.unique())
class_to_int = {c: i for i, c in enumerate(classes)}
int_to_class = {i: c for c, i in class_to_int.items()}
y_int = y.map(class_to_int).values

NUM_CLASSES = len(classes)
print("Classes detected:", classes)

# -----------------------------
# Identify categorical columns
# -----------------------------
cat_cols = [
    "age_group", "identity_code", "cultural_background",
    "upbringing_influence"
]

num_cols = [c for c in X.columns if c not in cat_cols + [ID_COL]]

# ============================================================
# Frequency Encoding
# ============================================================
def freq_encode(df1, df2, cols):
    for c in cols:
        freq = df1[c].value_counts()
        df1[c + "_FE"] = df1[c].map(freq)
        df2[c + "_FE"] = df2[c].map(freq)
    return df1, df2

X, T = freq_encode(X, T, cat_cols)

# ============================================================
# Target Mean Encoding (OOF, no leakage)
# ============================================================
def kfold_te(train_df, test_df, col, target):
    """
    Safe target encoding:
    - Uses only target values
    - No references to train_df[TARGET]
    - No leakage
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof = np.zeros(len(train_df))
    test_fold = np.zeros((len(test_df), 5))

    for fold, (tr, va) in enumerate(skf.split(train_df, target)):
        tr_x = train_df.iloc[tr]
        va_x = train_df.iloc[va]
        tr_y = target[tr]

        # Build mapping: value -> mean target
        df_tmp = pd.DataFrame({
            col: tr_x[col].values,
            "target": tr_y
        })

        mapping = df_tmp.groupby(col)["target"].mean()

        # Apply to validation fold
        oof[va] = va_x[col].map(mapping).fillna(mapping.mean())

        # Apply to test fold
        test_fold[:, fold] = test_df[col].map(mapping).fillna(mapping.mean())

    # Final TE column
    train_df[col + "_TE"] = oof
    test_df[col + "_TE"] = test_fold.mean(axis=1)

    return train_df, test_df


for c in cat_cols:
    X, T = kfold_te(X, T, c, y_int)

# ============================================================
# Scaling numeric features
# ============================================================
scaler = StandardScaler()
X_num = scaler.fit_transform(X[num_cols])
T_num = scaler.transform(T[num_cols])

# Final feature assembly
X_final = np.hstack([
    X_num,
    X[[c + "_FE" for c in cat_cols]].values,
    X[[c + "_TE" for c in cat_cols]].values
])

T_final = np.hstack([
    T_num,
    T[[c + "_FE" for c in cat_cols]].values,
    T[[c + "_TE" for c in cat_cols]].values
])

print("Final feature shapes:", X_final.shape, T_final.shape)

# ============================================================
# Compute class weights for XGBoost (major improvement)
# ============================================================
cw = compute_class_weight("balanced", classes=np.unique(y_int), y=y_int)
xgb_class_weights = {i: cw[i] for i in range(len(cw))}
print("XGB class weights:", xgb_class_weights)

# ============================================================
# Base models (6 strongest models only)
# ============================================================
def build_models():
    models = []

    # 1. Logistic Regression
    models.append(("LR", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="multinomial"
    )))

    # 2. SVM RBF (very strong)
    models.append(("SVM_RBF", SVC(
        kernel="rbf",
        C=2.5,
        gamma="scale",
        probability=True,
        class_weight="balanced"
    )))

    # 3. Linear SVM (calibrated)
    lin_svm = LinearSVC(C=1.0, class_weight="balanced")
    models.append(("SVM_LIN", CalibratedClassifierCV(lin_svm, cv=3)))

    # 4. Deep MLP
    models.append(("MLP", MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        learning_rate_init=0.0007,
        alpha=0.0005,
        max_iter=1200,
        batch_size=64,
        random_state=42
    )))

    # 5. XGB Deep
    models.append(("XGB_D", xgb.XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        tree_method="hist",
        eval_metric="mlogloss",
        scale_pos_weight=1.0,   # XGB handles weights internally for multi-class
        random_state=42
    )))

    # 6. XGB Medium
    models.append(("XGB_M", xgb.XGBClassifier(
        n_estimators=450,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.95,
        colsample_bytree=0.95,
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42
    )))

    return models


models = build_models()
M = len(models)
print("Using", M, "base models.")

# ============================================================
# First-level OOF Stacking
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

S_train = np.zeros((len(X_final), M * NUM_CLASSES))
S_test_temp = np.zeros((len(T_final), NUM_CLASSES, M))

for fold, (tr, va) in enumerate(skf.split(X_final, y_int)):
    print(f"\n===== FOLD {fold+1} =====")
    X_tr, X_va = X_final[tr], X_final[va]
    y_tr, y_va = y_int[tr], y_int[va]

    for idx, (name, model) in enumerate(models):
        print("Training:", name)

        model.fit(X_tr, y_tr)
        va_pred = model.predict_proba(X_va)

        f1 = f1_score(y_va, va_pred.argmax(axis=1), average="macro")
        print(f"{name} F1:", f1)

        S_train[va, idx*NUM_CLASSES:(idx+1)*NUM_CLASSES] = va_pred
        S_test_temp[:, :, idx] += model.predict_proba(T_final)

# average test preds
S_test = np.zeros((len(T_final), M * NUM_CLASSES))
for i in range(M):
    S_test[:, i*NUM_CLASSES:(i+1)*NUM_CLASSES] = S_test_temp[:, :, i] / 5

print("Meta features shape:", S_train.shape, S_test.shape)

# ============================================================
# Meta Model (XGB)
# ============================================================
meta_model = xgb.XGBClassifier(
    n_estimators=260,
    max_depth=4,
    learning_rate=0.04,
    subsample=0.95,
    colsample_bytree=0.95,
    objective="multi:softprob",
    num_class=NUM_CLASSES,
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=42
)

meta_model.fit(S_train, y_int)
oof_meta = meta_model.predict_proba(S_train)
print("Meta OOF F1:", f1_score(y_int, oof_meta.argmax(axis=1), average="macro"))

# ============================================================
# Seed Averaging for the Meta Model
# ============================================================
SEEDS = [42, 2024, 777, 909, 111, 222, 303, 404, 505, 606, 707, 808, 909, 1111, 2525]

final_pred = np.zeros((len(T_final), NUM_CLASSES))

for seed in SEEDS:
    model = xgb.XGBClassifier(
        n_estimators=260,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.95,
        colsample_bytree=0.95,
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=seed
    )
    model.fit(S_train, y_int)
    final_pred += model.predict_proba(S_test)

final_pred /= len(SEEDS)

# ============================================================
# Final decode + submission
# ============================================================
final_labels = final_pred.argmax(axis=1)
final_classes = [int_to_class[i] for i in final_labels]

submission = pd.DataFrame({
    "participant_id": test[ID_COL],
    "personality_cluster": final_classes
})

submission.to_csv("submission_superstack_15.csv", index=False)
print("\nSaved: submission_superstack_15.csv")
