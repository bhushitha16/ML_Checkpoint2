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
# Columns
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


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    print("Loading data...")
    
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # Target
    y = train[target_col].map({"Stayed": 1, "Left": 0})
    X = train[numerical_cols + categorical_cols]
    X_test_raw = test[numerical_cols + categorical_cols]

    # Preprocessing — EXACT v2
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    print("Preprocessing...")
    X_processed = preprocessor.fit_transform(X)
    X_test_processed = preprocessor.transform(X_test_raw)

    # Deterministic split (same as v2)
    RANDOM_SEED = 2024
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # ================================
    # Models (original v2 models)
    # ================================

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

    models["svm"] = CalibratedClassifierCV(
        svm_base, method="sigmoid", cv=3
    )

    models["gnb"] = GaussianNB()

    # Store predictions
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

        acc = accuracy_score(y_val, (val_pred >= 0.5).astype(int))
        print(f"{name} val accuracy: {acc:.5f}\n")

    # =======================================================
    # Weight Optimization (10,000 iterations)
    # =======================================================

    print("\n🔥 Running 10,000-iteration weight optimization...")
    
    model_names = list(models.keys())
    val_matrix = np.vstack([val_preds[m] for m in model_names])
    test_matrix = np.vstack([test_preds[m] for m in model_names])

    best_acc = 0
    best_w = None

    rng = np.random.default_rng(42)

    for i in range(10000):
        # Dirichlet sample — weights sum to 1
        w = rng.dirichlet(np.ones(len(model_names)))

        blended = np.dot(w, val_matrix)
        preds = (blended >= 0.5).astype(int)
        acc = accuracy_score(y_val, preds)

        if acc > best_acc:
            best_acc = acc
            best_w = w

        if (i + 1) % 1000 == 0:
            print(f"  Iter {i+1}/10000 | best OOF = {best_acc:.5f}")

    print("\n🔥 BEST ENSEMBLE OOF ACCURACY:", best_acc)
    print("🔥 BEST WEIGHTS:")
    for name, weight in zip(model_names, best_w):
        print(f"  {name}: {weight:.4f}")

    # =======================================================
    # Final blending on test set
    # =======================================================

    final_test_scores = np.dot(best_w, test_matrix)
    final_preds = ["Stayed" if p >= 0.5 else "Left" for p in final_test_scores]

    out_name = "submission_v8_optimized.csv"
    pd.DataFrame({
        id_col: test[id_col],
        target_col: final_preds
    }).to_csv(out_name, index=False)

    print(f"\n✅ Saved: {out_name}")


if __name__ == "__main__":
    main()
