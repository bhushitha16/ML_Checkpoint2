import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score


# ======================================================================
# Column definitions (same as your v2)
# ======================================================================

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


# ======================================================================
# Build Features (OneHot + Numeric FE)
# ======================================================================

def build_features(train, test):
    y = train[target_col].map({"Stayed": 1, "Left": 0}).values

    # ---------- Numeric handling ----------
    num_train = train[numerical_cols].copy()
    num_test = test[numerical_cols].copy()

    # Feature Engineering (light & stable)
    num_train["revenue_per_year"] = num_train["monthly_revenue_generated"] / (num_train["years_since_founding"] + 1)
    num_test["revenue_per_year"] = num_test["monthly_revenue_generated"] / (num_test["years_since_founding"] + 1)

    num_train["experience_ratio"] = num_train["founder_age"] / (num_train["years_with_startup"] + 1)
    num_test["experience_ratio"] = num_test["founder_age"] / (num_test["years_with_startup"] + 1)

    num_train["dependent_load"] = num_train["num_dependents"] * (num_train["years_with_startup"] + 1)
    num_test["dependent_load"] = num_test["num_dependents"] * (num_test["years_with_startup"] + 1)

    # Impute + scale
    num_imputer = SimpleImputer(strategy="median")
    num_scaler = StandardScaler()

    num_train_imp = num_imputer.fit_transform(num_train)
    num_test_imp = num_imputer.transform(num_test)

    num_train_scaled = num_scaler.fit_transform(num_train_imp)
    num_test_scaled = num_scaler.transform(num_test_imp)

    # ---------- Categorical handling ----------
    cat_train = train[categorical_cols].fillna("Missing")
    cat_test = test[categorical_cols].fillna("Missing")

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    ohe_train = ohe.fit_transform(cat_train)
    ohe_test = ohe.transform(cat_test)

    X = np.hstack([num_train_scaled, ohe_train])
    X_test = np.hstack([num_test_scaled, ohe_test])

    print("Final X shape:", X.shape)
    print("Final X_test shape:", X_test.shape)

    return X, X_test, y


# ======================================================================
# Model definitions
# ======================================================================

def make_logreg():
    return LogisticRegression(
        max_iter=800,
        C=0.4,
        class_weight="balanced",
        n_jobs=-1
    )

def make_ridge():
    return RidgeClassifierCV(
        alphas=[0.3, 0.6, 1.0],
        class_weight="balanced"
    )

def make_svm():
    base = LinearSVC(
        C=0.4,
        class_weight="balanced"
    )
    return CalibratedClassifierCV(
        estimator=base,
        method="sigmoid",
        cv=3
    )


# ======================================================================
# Main training with OOF stacking + weight search
# ======================================================================

def main():
    print("Loading data...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    X, X_test, y = build_features(train, test)

    n_samples = X.shape[0]
    n_test = X_test.shape[0]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_factories = {
        "logreg": make_logreg,
        "ridge": make_ridge,
        "svm": make_svm
    }

    oof_preds = {name: np.zeros(n_samples) for name in model_factories}
    test_preds_folds = {name: [] for name in model_factories}

    print("\n--- Training OOF models (5-fold) ---\n")
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold_idx+1}/5")

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        for name, factory in model_factories.items():
            print(f"  Training {name}...")
            model = factory()
            model.fit(X_tr, y_tr)

            # Predict proba
            if hasattr(model, "predict_proba"):
                val_p = model.predict_proba(X_val)[:, 1]
                test_p = model.predict_proba(X_test)[:, 1]
            else:
                val_p = model.decision_function(X_val)
                test_p = model.decision_function(X_test)

            oof_preds[name][val_idx] = val_p
            test_preds_folds[name].append(test_p)

        print()

    # Average test predictions across folds
    test_preds = {name: np.mean(np.vstack(test_preds_folds[name]), axis=0)
                  for name in model_factories}

    # ---------------------------------------------
    # Show OOF accuracies
    # ---------------------------------------------
    print("\nOOF performance per model:")
    for name in model_factories:
        preds = (oof_preds[name] >= 0.5).astype(int)
        acc = accuracy_score(y, preds)
        print(f"  {name}: {acc:.5f}")

    # ======================================================================
    # Ensemble weight optimization (Dirichlet search)
    # ======================================================================

    print("\nOptimizing ensemble weights...")

    names = list(model_factories.keys())
    oof_matrix = np.vstack([oof_preds[n] for n in names])
    test_matrix = np.vstack([test_preds[n] for n in names])

    best_acc = 0
    best_w = None

    rng = np.random.default_rng(42)

    for i in range(600):  # stable search
        w = rng.dirichlet(np.ones(len(names)))
        blended = np.dot(w, oof_matrix)
        preds = (blended >= 0.5).astype(int)
        acc = accuracy_score(y, preds)

        if acc > best_acc:
            best_acc = acc
            best_w = w

        if (i+1) % 150 == 0:
            print(f"  Iter {i+1}/600 | best OOF = {best_acc:.5f}")

    print("\nBest weights:")
    for n, wgt in zip(names, best_w):
        print(f"  {n}: {wgt:.4f}")

    print("Best OOF accuracy:", best_acc)

    # ======================================================================
    # Final test predictions
    # ======================================================================

    final_test = np.dot(best_w, test_matrix)
    final_labels = np.where(final_test >= 0.5, "Stayed", "Left")

    submission = pd.DataFrame({
        id_col: test[id_col],
        target_col: final_labels
    })

    out = "submission_simple_ensemble_v4.csv"
    submission.to_csv(out, index=False)
    print(f"\n✅ Saved: {out}")


# ======================================================================

if __name__ == "__main__":
    main()
