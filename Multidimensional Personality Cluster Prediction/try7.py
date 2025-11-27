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
import xgboost as xgb



# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

TARGET = "personality_cluster"
ID_COL = "participant_id"

y = train[TARGET]
X = train.drop(columns=[TARGET])
T = test.copy()



# ============================================================
# LABEL ENCODING FOR TARGET
# ============================================================

classes = sorted(y.unique())
class_to_int = {c:i for i,c in enumerate(classes)}
int_to_class = {i:c for c,i in class_to_int.items()}

print("Classes:", classes)
print("Encoded:", class_to_int)

y_int = y.map(class_to_int).values
NUM_CLASSES = len(classes)



# ============================================================
# FEATURE TYPE IDENTIFICATION
# ============================================================

cat_cols = [
    "age_group",
    "identity_code",
    "cultural_background",
    "upbringing_influence"
]

num_cols = [c for c in X.columns if c not in cat_cols + [ID_COL]]

print("\nCategorical columns:", cat_cols)
print("Numeric columns:", num_cols)



# ============================================================
# FREQUENCY ENCODING
# ============================================================

def freq_encode(train_df, test_df, cols):
    for col in cols:
        vc = train_df[col].value_counts()
        train_df[col + "_FE"] = train_df[col].map(vc)
        test_df[col + "_FE"] = test_df[col].map(vc)
    return train_df, test_df

X, T = freq_encode(X, T, cat_cols)



# ============================================================
# K-FOLD TARGET ENCODING (SAFE)
# ============================================================

def kfold_target_encode(train_df, test_df, col, target, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros(len(train_df))
    test_temp = np.zeros((len(test_df), n_splits))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, target)):

        tr_x = train_df.iloc[tr_idx]
        tr_y = target[tr_idx]
        val_x = train_df.iloc[val_idx]

        # Create a temporary df so groupby works
        df_tmp = pd.DataFrame({
            col: tr_x[col],
            "target": tr_y
        })

        mapping = df_tmp.groupby(col)["target"].mean()

        # Encode validation
        oof[val_idx] = val_x[col].map(mapping).fillna(mapping.mean())

        # Encode test
        test_temp[:, fold] = test_df[col].map(mapping).fillna(mapping.mean())

    # Average test encodings across folds
    test_encoded = test_temp.mean(axis=1)

    train_df[col + "_TE"] = oof
    test_df[col + "_TE"] = test_encoded

    return train_df, test_df




for col in cat_cols:
    X, T = kfold_target_encode(X, T, col, y_int)



# ============================================================
# RARITY ENCODING
# ============================================================

def rarity_encode(train_df, test_df, cols):
    for col in cols:
        vc = train_df[col].value_counts(normalize=True)
        train_df[col + "_RAR"] = train_df[col].map(vc)
        test_df[col + "_RAR"] = test_df[col].map(vc)
    return train_df, test_df

X, T = rarity_encode(X, T, cat_cols)



# ============================================================
# SCALE NUMERIC FEATURES
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[num_cols])
T_scaled = scaler.transform(T[num_cols])



# ============================================================
# FINAL STACKED FEATURE MATRIX
# ============================================================

feature_cols = (
    num_cols +
    [c + "_FE" for c in cat_cols] +
    [c + "_TE" for c in cat_cols] +
    [c + "_RAR" for c in cat_cols]
)

X_extra = X[[c for c in feature_cols if c not in num_cols]].values
T_extra = T[[c for c in feature_cols if c not in num_cols]].values

X_final = np.hstack([X_scaled, X_extra])
T_final = np.hstack([T_scaled, T_extra])

print("\nFinal Preprocessed Shapes:")
print("X_final:", X_final.shape)
print("T_final:", T_final.shape)



# ============================================================
# DEFINE BASE MODELS (CPU ONLY!)
# ============================================================

def build_models():
    models = []

    # Logistic Regression
    models.append(("LR", LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        solver="lbfgs",
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

    # MLP Neural Networks
    models.append(("MLP1", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )))

    models.append(("MLP2", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="tanh",
        learning_rate_init=0.001,
        max_iter=600,
        random_state=24
    )))

    # -------- XGBoost CPU MODELS --------
    # Deep
    models.append(("XGB_D1", xgb.XGBClassifier(
        n_estimators=650,
        max_depth=9,
        learning_rate=0.045,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="mlogloss"
    )))

    models.append(("XGB_D2", xgb.XGBClassifier(
        n_estimators=750,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        eval_metric="mlogloss"
    )))

    # Medium
    models.append(("XGB_M1", xgb.XGBClassifier(
        n_estimators=550,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="mlogloss"
    )))

    models.append(("XGB_M2", xgb.XGBClassifier(
        n_estimators=450,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.95,
        colsample_bytree=0.95,
        tree_method="hist",
        eval_metric="mlogloss"
    )))

    # Wide shallow
    models.append(("XGB_W1", xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="mlogloss"
    )))

    models.append(("XGB_W2", xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.10,
        subsample=0.95,
        colsample_bytree=0.95,
        tree_method="hist",
        eval_metric="mlogloss"
    )))

    return models



models = build_models()
M = len(models)

print("\n--------------------------------------------------")
print(" Total Base Models:", M)
print("--------------------------------------------------\n")



# ============================================================
# OOF STACKING
# ============================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

S_train = np.zeros((len(X_final), M * NUM_CLASSES))
S_test  = np.zeros((len(T_final), M * NUM_CLASSES))
test_accum = np.zeros((len(T_final), NUM_CLASSES, M))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_final, y_int)):
    print(f"\n===================== FOLD {fold+1} =====================")

    X_tr, X_val = X_final[tr_idx], X_final[val_idx]
    y_tr, y_val = y_int[tr_idx], y_int[val_idx]

    mi = 0

    for name, model in models:
        print(f"\n▶ Training {name}")

        model.fit(X_tr, y_tr)
        val_pred = model.predict_proba(X_val)

        f1 = f1_score(y_val, val_pred.argmax(axis=1), average="macro")
        print(f"{name} F1 = {f1:.4f}")

        S_train[val_idx, mi*NUM_CLASSES:(mi+1)*NUM_CLASSES] = val_pred
        test_accum[:, :, mi] += model.predict_proba(T_final)

        mi += 1



# Average test preds
for m in range(M):
    S_test[:, m*NUM_CLASSES:(m+1)*NUM_CLASSES] = test_accum[:, :, m] / skf.n_splits

print("\nMeta Feature Shapes:")
print("S_train:", S_train.shape)
print("S_test :", S_test.shape)



# ============================================================
# META-MODELS
# ============================================================

# ElasticNet Logistic Regression
meta_enet = LogisticRegression(
    max_iter=8000,
    penalty="elasticnet",
    solver="saga",
    l1_ratio=0.4,
    class_weight="balanced",
    random_state=42
)

meta_enet.fit(S_train, y_int)
enet_oof = meta_enet.predict_proba(S_train)
enet_f1 = f1_score(y_int, enet_oof.argmax(axis=1), average="macro")
print("\nElasticNet Meta OOF:", enet_f1)



# Tiny XGB meta model
meta_xgb = xgb.XGBClassifier(
    n_estimators=220,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.95,
    colsample_bytree=0.95,
    tree_method="hist",
    eval_metric="mlogloss",
)

meta_xgb.fit(S_train, y_int)
xgb_oof = meta_xgb.predict_proba(S_train)
xgb_f1 = f1_score(y_int, xgb_oof.argmax(axis=1), average="macro")
print("Tiny XGB Meta OOF:", xgb_f1)



# Blend meta model
blend_oof = 0.45 * enet_oof + 0.55 * xgb_oof
blend_f1 = f1_score(y_int, blend_oof.argmax(axis=1), average="macro")

print("\nBlended Meta OOF =", blend_f1)



# ============================================================
# SEED AVERAGING FOR FINAL PREDICTIONS
# ============================================================

SEEDS = [42,101,2024,777,888,909,2525,5050,303,999,111,808]

print("\nRunning Meta Seeds:")

final_pred = np.zeros((len(T_final), NUM_CLASSES))

def run_seed(seed):
    model = xgb.XGBClassifier(
        n_estimators=220,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.95,
        colsample_bytree=0.95,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=seed
    )
    model.fit(S_train, y_int)
    return model.predict_proba(S_test)


for s in SEEDS:
    print(" → Seed", s)
    final_pred += run_seed(s)

final_pred /= len(SEEDS)



# ============================================================
# FINAL SUBMISSION
# ============================================================

final_labels = final_pred.argmax(axis=1)
final_classes = [int_to_class[i] for i in final_labels]

submit = pd.DataFrame({
    "participant_id": test[ID_COL],
    "personality_cluster": final_classes
})

outfile = "submission_superstack_12_cpu.csv"
submit.to_csv(outfile, index=False)

print("\n====================================================")
print(" FINAL SUBMISSION SAVED:", outfile)
print("====================================================\n")
