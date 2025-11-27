# ============================================================
# SUPERSTACK 6.6 (NO CatBoost, NO LightGBM)
# Traditional ML Only — Target Macro-F1: 0.66 – 0.67
# PART 1 — ADVANCED PREPROCESSING + FEATURE ENGINEERING
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler, PowerTransformer, QuantileTransformer, PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Set global seed
# ------------------------------------------------------------
def seed_everything(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)

seed_everything(42)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "personality_cluster"
ID_COL = "participant_id"

y = train[TARGET]
X = train.drop(columns=[TARGET])
test_X = test.copy()

# Convert to int labels
classes = sorted(y.unique())
class_to_int = {c: i for i, c in enumerate(classes)}
int_to_class = {i: c for c, i in class_to_int.items()}
y_int = y.map(class_to_int).values
NUM_CLASSES = len(classes)

# ------------------------------------------------------------
# Columns
# ------------------------------------------------------------
cat_cols = ["age_group", "identity_code", "cultural_background", "upbringing_influence"]
num_cols = [
    "focus_intensity",
    "consistency_score",
    "external_guidance_usage",
    "support_environment_score",
    "hobby_engagement_level",
    "physical_activity_index",
    "creative_expression_index",
    "altruism_score"
]

# ------------------------------------------------------------
# FREQUENCY ENCODING
# ------------------------------------------------------------
def frequency_encode(train_df, test_df, cols):
    for c in cols:
        freq = train_df[c].value_counts()
        train_df[c + "_FE"] = train_df[c].map(freq)
        test_df[c + "_FE"] = test_df[c].map(freq)
    return train_df, test_df

X, test_X = frequency_encode(X, test_X, cat_cols)

# ------------------------------------------------------------
# TARGET ENCODING (OOF, per-class, smoothed)
# ------------------------------------------------------------
def target_encode_smooth_oof(train_df, test_df, col, y, n_splits=5, alpha=20):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros(len(train_df))
    test_encoded = np.zeros(len(test_df))

    global_mean = y.mean()

    for tr_idx, val_idx in skf.split(train_df, y):
        tr = train_df.iloc[tr_idx].copy()
        val = train_df.iloc[val_idx].copy()

        tr["_y"] = y[tr_idx]
        means = tr.groupby(col)["_y"].mean()

        # smoothing
        counts = tr[col].value_counts()
        smooth = ((means * counts) + (global_mean * alpha)) / (counts + alpha)

        oof[val_idx] = val[col].map(smooth).fillna(global_mean)
        test_encoded += test_df[col].map(smooth).fillna(global_mean) / n_splits

    train_df[col + "_TE"] = oof
    test_df[col + "_TE"] = test_encoded
    return train_df, test_df

for c in cat_cols:
    X, test_X = target_encode_smooth_oof(X, test_X, c, y_int)

# ------------------------------------------------------------
# ORDINAL ENCODING
# ------------------------------------------------------------
ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_OE = ord_encoder.fit_transform(X[cat_cols])
test_OE = ord_encoder.transform(test_X[cat_cols])

# ------------------------------------------------------------
# ONE-HOT ENCODING
# ------------------------------------------------------------
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_OHE = ohe.fit_transform(X[cat_cols])
test_OHE = ohe.transform(test_X[cat_cols])

# ------------------------------------------------------------
# POLYNOMIAL INTERACTIONS FOR NUMERIC
# ------------------------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X[num_cols])
test_poly = poly.transform(test_X[num_cols])

# ------------------------------------------------------------
# NUMERIC TRANSFORMATIONS (3 types)
# ------------------------------------------------------------
rs = RobustScaler()
qt = QuantileTransformer(output_distribution="normal")
pt = PowerTransformer()

X_RS = rs.fit_transform(X[num_cols])
test_RS = rs.transform(test_X[num_cols])

X_QT = qt.fit_transform(X[num_cols])
test_QT = qt.transform(test_X[num_cols])

X_PT = pt.fit_transform(X[num_cols])
test_PT = pt.transform(test_X[num_cols])

# ------------------------------------------------------------
# CATEGORICAL INTERACTIONS (SUPER IMPORTANT)
# ------------------------------------------------------------
def cat_interactions(df, test_df, cols):
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            name = f"{cols[i]}_{cols[j]}"
            df[name] = df[cols[i]].astype(str) + "_" + df[cols[j]].astype(str)
            test_df[name] = test_df[cols[i]].astype(str) + "_" + test_df[cols[j]].astype(str)
    return df, test_df

X, test_X = cat_interactions(X, test_X, cat_cols)

# Ordinal encode these interactions
inter_cols = [c for c in X.columns if "_" in c and c.endswith(tuple([col for col in cat_cols])) is False]

inter_ord = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_INT = inter_ord.fit_transform(X[inter_cols])
test_INT = inter_ord.transform(test_X[inter_cols])

# ------------------------------------------------------------
# BUILD FINAL FEATURE MATRIX
# ------------------------------------------------------------
def build_final_features():
    X_final = np.hstack([
        X_OE,          # ordinal cat
        X_OHE,         # one-hot cat
        X[ [c + "_FE" for c in cat_cols] ].values,   # frequency
        X[ [c + "_TE" for c in cat_cols] ].values,   # target encoding
        X_RS, X_QT, X_PT,     # numeric transforms
        X_poly,               # polynomial interactions
        X_INT                 # categorical interactions
    ])

    T_final = np.hstack([
        test_OE,
        test_OHE,
        test_X[ [c + "_FE" for c in cat_cols] ].values,
        test_X[ [c + "_TE" for c in cat_cols] ].values,
        test_RS, test_QT, test_PT,
        test_poly,
        test_INT
    ])

    return X_final, T_final

X_final, T_final = build_final_features()

print("Final Feature Matrix Shapes:")
print("X_final:", X_final.shape)
print("T_final:", T_final.shape)

# ------------------------------------------------------------
# OOF SPLITTER
# ------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# ============================================================
# SUPERSTACK 6.6 — PART 2
# Base Models + XGBoost v3.1.1 + OOF Stacking
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# ------------------------------------------------------------
# BASE MODELS (ONLY ALLOWED MODELS)
# ------------------------------------------------------------
def get_base_models():
    models = []

    # 1. LOGISTIC REGRESSION (balanced)
    models.append(("LOGR", LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="multinomial"
    )))

    # 2. L1 Logistic
    models.append(("LOGR_L1", LogisticRegression(
        max_iter=4000,
        penalty="l1",
        solver="saga",
        class_weight="balanced"
    )))

    # 3. L2 Logistic
    models.append(("LOGR_L2", LogisticRegression(
        max_iter=4000,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced"
    )))

    # 4. ElasticNet Logistic Regression
    models.append(("LOGR_EN", LogisticRegression(
        max_iter=4000,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        class_weight="balanced"
    )))

    # 5. SVM (balanced)
    models.append(("SVM", SVC(
        probability=True,
        kernel="rbf",
        C=2.0,
        gamma="scale",
        class_weight="balanced"
    )))

    # 6. MLP Neural Network
    models.append(("MLP", MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=500,
        learning_rate_init=0.0008,
        random_state=42
    )))

    # 7. Random Forest
    models.append(("RF", RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_split=3,
        class_weight="balanced",
        random_state=42
    )))

    # 8. Extra Trees
    models.append(("ET", ExtraTreesClassifier(
        n_estimators=400,
        max_depth=15,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42
    )))

    return models


# ------------------------------------------------------------
# XGBOOST VARIANTS (optimized for v3.1.1)
# ------------------------------------------------------------
def get_xgboost_models():
    models = []

    # XGB 1 — Deep trees
    models.append(("XGB_DEEP", xgb.XGBClassifier(
        n_estimators=650,
        max_depth=9,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        device="cpu",
        tree_method="hist",
        random_state=42
    )))

    # XGB 2 — Wide trees
    models.append(("XGB_WIDE", xgb.XGBClassifier(
        n_estimators=750,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        device="cpu",
        tree_method="hist",
        random_state=42
    )))

    # XGB 3 — Regularized trees
    models.append(("XGB_REG", xgb.XGBClassifier(
        n_estimators=500,
        max_depth=7,
        reg_lambda=2.0,
        reg_alpha=1.0,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        device="cpu",
        tree_method="hist",
        random_state=42
    )))

    return models


# ------------------------------------------------------------
# OOF STACKING (ALL BASE MODELS + XGBOOST ×3)
# ------------------------------------------------------------
def run_oof_stacking_all(Xf, y_int, Tf):
    models = get_base_models() + get_xgboost_models()
    M = len(models)
    print("\nTotal base models:", M)

    # Out-of-fold meta features
    S_train = np.zeros((len(Xf), M * NUM_CLASSES))
    S_test = np.zeros((len(Tf), M * NUM_CLASSES))

    # Per-model test storage (for averaging within folds)
    test_preds = np.zeros((len(Tf), NUM_CLASSES, M))

    # Start stacking
 # ============================================================
# SUPERSTACK 6.6 — PART 2 (FIXED OOF STACKING)
# ============================================================

def run_oof_stacking_all(Xf, y_int, Tf):
    models = get_base_models() + get_xgboost_models()
    M = len(models)
    print("\nTotal base models:", M)

    # meta feature matrices
    S_train = np.zeros((len(Xf), M * NUM_CLASSES))
    S_test  = np.zeros((len(Tf), M * NUM_CLASSES))

    # test preds storage across folds
    test_fold_preds = np.zeros((len(Tf), NUM_CLASSES, M))

    for fold, (train_idx, val_idx) in enumerate(skf.split(Xf, y_int)):
        print(f"\n========== FOLD {fold+1} ==========")

        X_train_fold, X_val_fold = Xf[train_idx], Xf[val_idx]
        y_train_fold, y_val_fold = y_int[train_idx], y_int[val_idx]

        model_index = 0  # IMPORTANT — RESET per fold

        for name, model in models:
            print(f"\nTraining model: {name}")

            # Train
            model.fit(X_train_fold, y_train_fold)

            # Validation predictions
            val_pred = model.predict_proba(X_val_fold)
            f1 = f1_score(y_val_fold, val_pred.argmax(axis=1), average="macro")
            print(f"{name} Fold F1 = {f1:.4f}")

            # store OOF predictions
            start = model_index * NUM_CLASSES
            end   = (model_index + 1) * NUM_CLASSES
            S_train[val_idx, start:end] = val_pred

            # store test preds for averaging
            test_fold_preds[:, :, model_index] += model.predict_proba(Tf)

            model_index += 1

    # average test predictions across folds
    for m in range(M):
        S_test[:, m*NUM_CLASSES:(m+1)*NUM_CLASSES] = test_fold_preds[:, :, m] / skf.get_n_splits()

    return S_train, S_test


# ============================================================
# SUPERSTACK 6.6 — PART 3 (META MODEL + SEED AVERAGING)
# ============================================================

from sklearn.linear_model import LogisticRegression

print("\n====================================================")
print(" Running OOF Stacking for All Base Models + XGBoost ")
print("====================================================\n")

S_train, S_test = run_oof_stacking_all(X_final, y_int, T_final)

print("\nMeta Feature Shapes:")
print("S_train:", S_train.shape)
print("S_test :", S_test.shape)


# ------------------------------------------------------------
# LEVEL-2 META MODEL (ElasticNet Logistic Regression)
# ------------------------------------------------------------
print("\n====================================================")
print(" Training ENET Logistic Regression Meta Model")
print("====================================================\n")

meta_model = LogisticRegression(
    max_iter=6000,
    solver="saga",
    penalty="elasticnet",
    l1_ratio=0.5,
    class_weight="balanced",
    random_state=42
)

meta_model.fit(S_train, y_int)

oof_meta = meta_model.predict_proba(S_train)
oof_f1 = f1_score(y_int, oof_meta.argmax(axis=1), average="macro")

print(f"\nMeta Model OOF Macro-F1 = {oof_f1:.4f}\n")


# ------------------------------------------------------------
# FINAL 7-SEED AVERAGING
# ------------------------------------------------------------
SEEDS = [42, 2024, 101, 777, 999, 2525, 5050]

def run_meta_with_seed(seed):
    model = LogisticRegression(
        max_iter=6000,
        solver="saga",
        penalty="elasticnet",
        l1_ratio=0.5,
        class_weight="balanced",
        random_state=seed
    )
    model.fit(S_train, y_int)
    return model.predict_proba(S_test)

print("\n==============================")
print(" Running 7-Seed Averaging...")
print("==============================\n")

final_pred = np.zeros((len(S_test), NUM_CLASSES))

for s in SEEDS:
    print(f" → Executing seed {s}...")
    final_pred += run_meta_with_seed(s)

final_pred /= len(SEEDS)


# ------------------------------------------------------------
# DECODE FINAL LABELS + SAVE SUBMISSION
# ------------------------------------------------------------
final_labels = final_pred.argmax(axis=1)
final_classes = [int_to_class[i] for i in final_labels]

submission = pd.DataFrame({
    "participant_id": test[ID_COL],
    "personality_cluster": final_classes
})

output_file = "submission_superstack_6_6.csv"
submission.to_csv(output_file, index=False)

print("\n====================================================")
print(" FINAL SUBMISSION SAVED:", output_file)
print("====================================================\n")
