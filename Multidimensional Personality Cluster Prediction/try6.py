# ============================================================
# SUPERSTACK 11.0 — CPU OPTIMIZED
# Preprocessing + Target Encoding + Base Models + Meta Layer
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

TARGET = "personality_cluster"
ID_COL = "participant_id"

y = train[TARGET]
X = train.drop(columns=[TARGET])
T = test.copy()

# Class Encoding
classes = sorted(y.unique())
class_to_int = {c:i for i,c in enumerate(classes)}
int_to_class = {i:c for c,i in class_to_int.items()}
y_int = y.map(class_to_int).values
NUM_CLASSES = len(classes)

# ------------------------------------------------------------
# 2. COLUMN GROUPS
# ------------------------------------------------------------
cat_cols = ["age_group", "identity_code", "cultural_background", "upbringing_influence"]

num_cols = [c for c in X.columns if c not in cat_cols + [ID_COL]]

# ------------------------------------------------------------
# 3. FREQUENCY ENCODING
# ------------------------------------------------------------
def frequency_encode(df_tr, df_te, cols):
    for col in cols:
        freq = df_tr[col].value_counts()
        df_tr[col+"_FE"] = df_tr[col].map(freq)
        df_te[col+"_FE"] = df_te[col].map(freq)
    return df_tr, df_te

X, T = frequency_encode(X, T, cat_cols)

# ------------------------------------------------------------
# 4. TARGET MEAN ENCODING (SAFE KFOLD)
# ------------------------------------------------------------
def kfold_target_encode(df_tr, df_te, col, target, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros(len(df_tr))
    test_temp = np.zeros((len(df_te), n_splits))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(df_tr, target)):
        tr = df_tr.iloc[tr_idx]
        val = df_tr.iloc[val_idx]
        tr_y = target[tr_idx]

        grp = pd.DataFrame({col: tr[col], "y": tr_y})
        mapping = grp.groupby(col)["y"].mean()

        oof[val_idx] = val[col].map(mapping).fillna(mapping.mean())
        test_temp[:, fold] = df_te[col].map(mapping).fillna(mapping.mean())

    df_tr[col+"_TE"] = oof
    df_te[col+"_TE"] = test_temp.mean(axis=1)

    return df_tr, df_te

for c in cat_cols:
    X, T = kfold_target_encode(X, T, c, y_int)

# ------------------------------------------------------------
# 5. RARITY ENCODING
# ------------------------------------------------------------
def rarity_encode(df_tr, df_te, cols):
    for col in cols:
        vc = df_tr[col].value_counts(normalize=True)
        df_tr[col+"_RAR"] = df_tr[col].map(vc)
        df_te[col+"_RAR"] = df_te[col].map(vc)
    return df_tr, df_te

X, T = rarity_encode(X, T, cat_cols)

# ------------------------------------------------------------
# 6. SCALING NUMERIC FEATURES
# ------------------------------------------------------------
scaler = StandardScaler()
X_num = scaler.fit_transform(X[num_cols])
T_num = scaler.transform(T[num_cols])

# Final stacks
X_final = np.hstack([
    X_num,
    X[[c+"_FE" for c in cat_cols]].values,
    X[[c+"_TE" for c in cat_cols]].values,
    X[[c+"_RAR" for c in cat_cols]].values
])

T_final = np.hstack([
    T_num,
    T[[c+"_FE" for c in cat_cols]].values,
    T[[c+"_TE" for c in cat_cols]].values,
    T[[c+"_RAR" for c in cat_cols]].values
])

print("\nFinal Preprocessed Shapes:")
print("X_final:", X_final.shape)
print("T_final:", T_final.shape)

# ============================================================
# 7. DEFINE BASE MODELS (LR + SVM + NN + XGB)
# ============================================================

def build_models():
    models = []

    # Logistic Regression
    models.append(("LR", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        multi_class="multinomial"
    )))

    # SVM
    models.append(("SVM", SVC(
        probability=True,
        kernel="rbf",
        C=1.2,
        gamma="scale",
        class_weight="balanced"
    )))

    # MLP Neural Network
    models.append(("MLP", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        learning_rate_init=0.001,
        max_iter=600,
        random_state=42
    )))

    # XGBoost Deep
    models.append(("XGB_D1", xgb.XGBClassifier(
        n_estimators=600,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist"
    )))

    models.append(("XGB_D2", xgb.XGBClassifier(
        n_estimators=700,
        max_depth=10,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist"
    )))

    # XGBoost Medium
    models.append(("XGB_M1", xgb.XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist"
    )))

    models.append(("XGB_M2", xgb.XGBClassifier(
        n_estimators=450,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.95,
        colsample_bytree=0.95,
        eval_metric="mlogloss",
        tree_method="hist"
    )))

    # XGBoost Wide (shallow but strong)
    models.append(("XGB_W1", xgb.XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.09,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist"
    )))

    models.append(("XGB_W2", xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.10,
        subsample=0.95,
        colsample_bytree=0.95,
        eval_metric="mlogloss",
        tree_method="hist"
    )))

    return models

models = build_models()
M = len(models)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nTotal Base Models:", M)

# Allocate OOF matrices
S_train = np.zeros((len(X_final), M * NUM_CLASSES))
S_test  = np.zeros((len(T_final), M * NUM_CLASSES))
test_acc = np.zeros((len(T_final), NUM_CLASSES, M))

# ============================================================
# 8. GENERATE OOF META FEATURES
# ============================================================
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_final, y_int)):
    print(f"\n===================== FOLD {fold+1} =====================\n")

    X_tr, X_val = X_final[tr_idx], X_final[val_idx]
    y_tr, y_val = y_int[tr_idx], y_int[val_idx]

    model_idx = 0

    for name, model in models:
        print(f"▶ Training {name}")

        model.fit(X_tr, y_tr)
        val_pred = model.predict_proba(X_val)

        f1 = f1_score(y_val, val_pred.argmax(axis=1), average="macro")
        print(f"{name} F1 = {f1:.4f}\n")

        S_train[val_idx, model_idx*NUM_CLASSES:(model_idx+1)*NUM_CLASSES] = val_pred
        test_acc[:, :, model_idx] += model.predict_proba(T_final)

        model_idx += 1

# Average test predictions
for m in range(M):
    S_test[:, m*NUM_CLASSES:(m+1)*NUM_CLASSES] = test_acc[:, :, m] / skf.n_splits

print("\nMeta Feature Shapes:")
print("S_train:", S_train.shape)
print("S_test :", S_test.shape)

# ============================================================
# 9. META-MODEL (DUAL)
# ============================================================

# ElasticNet Logistic Regression
meta_en = LogisticRegression(
    max_iter=6000,
    solver="saga",
    penalty="elasticnet",
    l1_ratio=0.4,
    class_weight="balanced"
)

meta_en.fit(S_train, y_int)
en_oof = meta_en.predict_proba(S_train)
en_f1 = f1_score(y_int, en_oof.argmax(axis=1), average="macro")
print("\nElasticNet Meta OOF:", en_f1)

# Tiny XGBoost as Meta-model
meta_xgb = xgb.XGBClassifier(
    n_estimators=180,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.95,
    colsample_bytree=0.95,
    eval_metric="mlogloss",
    tree_method="hist"
)

meta_xgb.fit(S_train, y_int)
xgb_oof = meta_xgb.predict_proba(S_train)
xgb_f1 = f1_score(y_int, xgb_oof.argmax(axis=1), average="macro")
print("Tiny XGB Meta OOF:", xgb_f1)

# Blended Meta
final_oof = 0.45 * en_oof + 0.55 * xgb_oof
blend_oof_f1 = f1_score(y_int, final_oof.argmax(axis=1), average="macro")
print("\nBlended Meta OOF =", blend_oof_f1)

# ============================================================
# 10. SEED AVERAGING
# ============================================================

SEEDS = [42,101,2024,777,888,909,2525,5050,303,999,111,808]
final_pred = np.zeros((len(T_final), NUM_CLASSES))

def run_meta(seed):
    model = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.95,
        colsample_bytree=0.95,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed
    )
    model.fit(S_train, y_int)
    return model.predict_proba(S_test)

print("\nRunning Meta Seeds:")
for s in SEEDS:
    print(" → Seed", s)
    final_pred += run_meta(s)

final_pred /= len(SEEDS)

# ============================================================
# 11. CREATE SUBMISSION
# ============================================================

final_labels = final_pred.argmax(axis=1)
final_classes = [int_to_class[i] for i in final_labels]

submission = pd.DataFrame({
    "participant_id": test[ID_COL],
    "personality_cluster": final_classes
})

outfile = "submission_superstack_11_cpu.csv"
submission.to_csv(outfile, index=False)

print("\n====================================================")
print(" FINAL SUBMISSION SAVED:", outfile)
print("====================================================\n")
