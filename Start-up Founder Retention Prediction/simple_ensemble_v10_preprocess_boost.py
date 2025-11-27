import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# ============================================================
# Columns (same as v2/v7)
# ============================================================
numerical_cols = [
    'founder_age', 'years_with_startup', 'monthly_revenue_generated',
    'funding_rounds_led', 'distance_from_investor_hub',
    'num_dependents', 'years_since_founding'
]

categorical_cols = [
    'founder_gender', 'founder_role', 'work_life_balance_rating',
    'venture_satisfaction', 'startup_performance_rating', 'working_overtime',
    'education_background', 'personal_status', 'startup_stage',
    'team_size_category', 'remote_operations', 'leadership_scope',
    'innovation_support', 'startup_reputation', 'founder_visibility'
]

target_col = "retention_status"
id_col = "founder_id"

# 50 seeds list (same as v9)
SEED_LIST = [
   8,9,23,29,4044,5055
]


# ============================================================
# Run single seed (same models + weights, new preprocessing)
# ============================================================
def run_seed(seed):
    print(f"\n==============================")
    print(f"Running seed = {seed}")
    print(f"==============================")

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # --------------------------------------------------------
    # 1) SORT TRAINING DATA — improves SVM consistency
    # --------------------------------------------------------
    train = train.sort_values(by=id_col).reset_index(drop=True)

    # --------------------------------------------------------
    # 2) ADD MISSING INDICATOR FEATURES (numerical)
    # --------------------------------------------------------
    for col in numerical_cols:
        train[col + "_missing"] = train[col].isna().astype(int)
        test[col + "_missing"] = test[col].isna().astype(int)

    # update column lists
    missing_indicator_cols = [col + "_missing" for col in numerical_cols]
    all_numerical_cols = numerical_cols + missing_indicator_cols

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------
    y = train[target_col].map({"Stayed": 1, "Left": 0})
    X = train[all_numerical_cols + categorical_cols]
    X_test_raw = test[all_numerical_cols + categorical_cols]

    # --------------------------------------------------------
    # 3) REPLACE CATEGORICAL IMPUTATION WITH CONSTANT "Unknown"
    # --------------------------------------------------------
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="constant", fill_value="Unknown")),
        ('onehot', OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, all_numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    X_test_processed = preprocessor.transform(X_test_raw)

    # --------------------------------------------------------
    # Train/Val Split (same logic)
    # --------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.20, random_state=seed, stratify=y
    )

    # --------------------------------------------------------
    # MODELS — EXACT v2/v7
    # --------------------------------------------------------
    models = {}

    models["logreg"] = LogisticRegression(
        max_iter=2000,
        C=0.3,
        class_weight="balanced",
        n_jobs=-1
    )

    models["ridge"] = RidgeClassifierCV(
        alphas=[0.1, 0.5, 1.0, 2.0],
        class_weight="balanced"
    )

    svm_base = LinearSVC(
        C=0.5,
        class_weight="balanced",
        max_iter=5000
    )
    models["svm"] = CalibratedClassifierCV(svm_base, method="sigmoid", cv=3)

    models["gnb"] = GaussianNB()

    # --------------------------------------------------------
    # TRAIN + STORE PREDS
    # --------------------------------------------------------
    val_preds = {}
    test_preds = {}

    print("\nTraining models...\n")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            val_pred = model.predict_proba(X_val)[:, 1]
            test_pred = model.predict_proba(X_test_processed)[:, 1]
        else:
            val_pred = model.decision_function(X_val)
            test_pred = model.decision_function(X_test_processed)

        val_preds[name] = val_pred
        test_preds[name] = test_pred

        print(f"{name} Val Accuracy: {accuracy_score(y_val, (val_pred>=0.5).astype(int)):.5f}")

    # --------------------------------------------------------
    # ORIGINAL v2 WEIGHTS — DO NOT CHANGE
    # --------------------------------------------------------
    weights = {
        "logreg": 0.32,
        "ridge": 0.34,
        "svm": 0.28,
        "gnb": 0.06
    }

    # --------------------------------------------------------
    # ENSEMBLE VAL + TEST PREDICTIONS
    # --------------------------------------------------------
    val_ensemble = np.zeros_like(list(val_preds.values())[0])
    for name in models:
        val_ensemble += weights[name] * val_preds[name]

    val_acc = accuracy_score(y_val, (val_ensemble >= 0.5).astype(int))
    print(f"\n🔥 Ensemble Validation Accuracy (seed {seed}): {val_acc:.5f}")

    # Test predictions
    test_ensemble = np.zeros_like(list(test_preds.values())[0])
    for name in models:
        test_ensemble += weights[name] * test_preds[name]

    final_labels = ["Stayed" if p >= 0.5 else "Left" for p in test_ensemble]

    fname = f"submission_v10_seed_{seed}.csv"
    pd.DataFrame({
        id_col: test[id_col],
        target_col: final_labels
    }).to_csv(fname, index=False)

    print(f"Saved: {fname}")

    return seed, val_acc



# ============================================================
# Run all 50 seeds (same as v9)
# ============================================================
if __name__ == "__main__":
    results = []

    for seed in SEED_LIST:
        s, acc = run_seed(seed)
        results.append((s, acc))

    # Sort by best val accuracy
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n==================== FINAL SEED RANKING ====================")
    for seed, acc in results:
        print(f"Seed {seed}: Val Accuracy = {acc:.5f}")

    print("\n🔥 Upload top 5 seeds – these have real chance of beating 0.754!")
