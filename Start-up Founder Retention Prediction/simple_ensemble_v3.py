import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

# =========================================================
# Column definitions (same as your v2)
# =========================================================

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


# =========================================================
# Utility: target encoding (KFold, leakage-safe)
# =========================================================

def kfold_target_encode(train_series, y, test_series, folds, smoothing=1.0):
    """
    Returns:
        train_te: np.array of encoded values for train
        test_te: np.array of encoded values for test
    """
    train_te = np.zeros(len(train_series), dtype=float)
    test_te = np.zeros(len(test_series), dtype=float)

    global_mean = y.mean()

    train_df = pd.DataFrame({
        "cat": train_series,
        "target": y
    })

    # OOF target encoding for train
    for tr_idx, val_idx in folds:
        tr = train_df.iloc[tr_idx]
        means = tr.groupby("cat")["target"].mean()

        # Optional smoothing
        counts = tr.groupby("cat")["target"].count()
        smooth = (means * counts + global_mean * smoothing) / (counts + smoothing)

        mapped = train_series.iloc[val_idx].map(smooth)
        train_te[val_idx] = mapped.fillna(global_mean).values

    # Full-data encoding for test
    full_means = train_df.groupby("cat")["target"].mean()
    full_counts = train_df.groupby("cat")["target"].count()
    full_smooth = (full_means * full_counts + global_mean * smoothing) / (full_counts + smoothing)
    mapped_test = test_series.map(full_smooth)
    test_te = mapped_test.fillna(global_mean).values

    return train_te, test_te


# =========================================================
# Hybrid preprocessing: numeric + OHE + target + freq
# =========================================================

def build_features(train, test):
    """
    Build a strong feature matrix for this problem.
    Returns:
        X (np.ndarray), X_test (np.ndarray), y (np.ndarray)
    """
    # Map target
    y = train[target_col].map({"Stayed": 1, "Left": 0}).values

    # ------- numerical base -------
    num_train = train[numerical_cols].copy()
    num_test = test[numerical_cols].copy()

    # ------- simple numeric feature engineering -------
    # Safe operations with +1 to avoid division by zero
    # Adjust if any column names differ
    num_train["revenue_per_year"] = num_train["monthly_revenue_generated"] / (num_train["years_since_founding"] + 1)
    num_test["revenue_per_year"] = num_test["monthly_revenue_generated"] / (num_test["years_since_founding"] + 1)

    num_train["experience_ratio"] = num_train["founder_age"] / (num_train["years_with_startup"] + 1)
    num_test["experience_ratio"] = num_test["founder_age"] / (num_test["years_with_startup"] + 1)

    num_train["dependents_ratio"] = num_train["num_dependents"] / (num_train["founder_age"] + 1)
    num_test["dependents_ratio"] = num_test["num_dependents"] / (num_test["founder_age"] + 1)

    # ------- impute + scale numeric -------
    num_imputer = SimpleImputer(strategy="median")
    num_scaler = StandardScaler()

    num_train_imputed = num_imputer.fit_transform(num_train)
    num_test_imputed = num_imputer.transform(num_test)

    num_train_scaled = num_scaler.fit_transform(num_train_imputed)
    num_test_scaled = num_scaler.transform(num_test_imputed)

    # ------- Categorical handling strategy -------
    cat_train = train[categorical_cols].copy()
    cat_test = test[categorical_cols].copy()

    # Fill NaNs with explicit placeholder for categorical stuff
    cat_train = cat_train.fillna("Missing")
    cat_test = cat_test.fillna("Missing")

    # Decide which columns are low / mid / high cardinality
    nunique = cat_train.nunique()

    LOW_MAX = 6      # <=6 → OneHot
    MID_MAX = 20     # 7–20 → Target encoding
    # >20 → Frequency encoding

    low_card_cols = [c for c in categorical_cols if nunique[c] <= LOW_MAX]
    mid_card_cols = [c for c in categorical_cols if LOW_MAX < nunique[c] <= MID_MAX]
    high_card_cols = [c for c in categorical_cols if nunique[c] > MID_MAX]

    print("Low-cardinality (OneHot):", low_card_cols)
    print("Mid-cardinality (Target Encoded):", mid_card_cols)
    print("High-cardinality (Frequency Encoded):", high_card_cols)

    # ------- OneHot for low-cardinality -------
    if low_card_cols:
        combined_low = pd.concat(
            [cat_train[low_card_cols], cat_test[low_card_cols]],
            axis=0,
            ignore_index=True
        )
        combined_low = pd.get_dummies(combined_low, columns=low_card_cols, dummy_na=True)
        ohe_train = combined_low.iloc[:len(train)].values
        ohe_test = combined_low.iloc[len(train):].values
    else:
        ohe_train = np.zeros((len(train), 0))
        ohe_test = np.zeros((len(test), 0))

    # Prepare folds once for target encoding (reused later for model OOF too)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(train, y))

    # ------- Target encoding for mid-cardinality -------
    te_train_list = []
    te_test_list = []

    for col in mid_card_cols:
        tr_vals, te_vals = kfold_target_encode(
            train_series=cat_train[col],
            y=y,
            test_series=cat_test[col],
            folds=folds,
            smoothing=1.0
        )
        te_train_list.append(tr_vals.reshape(-1, 1))
        te_test_list.append(te_vals.reshape(-1, 1))

    if te_train_list:
        te_train = np.hstack(te_train_list)
        te_test = np.hstack(te_test_list)
    else:
        te_train = np.zeros((len(train), 0))
        te_test = np.zeros((len(test), 0))

    # ------- Frequency encoding for high-cardinality -------
    fe_train_list = []
    fe_test_list = []

    for col in high_card_cols:
        freqs = cat_train[col].value_counts(normalize=True)
        tr_freq = cat_train[col].map(freqs).fillna(0).values.reshape(-1, 1)
        ts_freq = cat_test[col].map(freqs).fillna(0).values.reshape(-1, 1)
        fe_train_list.append(tr_freq)
        fe_test_list.append(ts_freq)

    if fe_train_list:
        fe_train = np.hstack(fe_train_list)
        fe_test = np.hstack(fe_test_list)
    else:
        fe_train = np.zeros((len(train), 0))
        fe_test = np.zeros((len(test), 0))

    # ------- Final X matrices -------
    X = np.hstack([num_train_scaled, ohe_train, te_train, fe_train])
    X_test = np.hstack([num_test_scaled, ohe_test, te_test, fe_test])

    print("Final train feature shape:", X.shape)
    print("Final test feature shape:", X_test.shape)

    return X, X_test, y, folds


# =========================================================
# Build and train OOF models + ensemble
# =========================================================

def main():
    print("Loading data...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    X, X_test, y, folds = build_features(train, test)

    n_samples = X.shape[0]
    n_test = X_test.shape[0]

    # Define models (each is a factory to get a FRESH instance per fold)
    def make_logreg():
        return LogisticRegression(
            max_iter=1000,
            C=0.5,
            class_weight="balanced",
            n_jobs=-1
        )

    def make_svm():
        base = LinearSVC(
            C=0.5,
            class_weight="balanced"
        )
        # Calibrated to get probabilities
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)

    def make_gnb():
        return GaussianNB()

    model_factories = {
        "logreg": make_logreg,
        "svm": make_svm,
        "gnb": make_gnb
    }

    # Storage for OOF and test predictions
    oof_preds = {name: np.zeros(n_samples, dtype=float) for name in model_factories}
    test_preds_folds = {name: [] for name in model_factories}

    print("\nTraining models with 5-fold OOF...\n")

    # Use the same folds as used for target encoding
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        print(f"Fold {fold_idx + 1}/{len(folds)}")

        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        for name, factory in model_factories.items():
            print(f"  - Training {name}...")
            model = factory()
            model.fit(X_tr, y_tr)

            # Predict on validation
            if hasattr(model, "predict_proba"):
                val_proba = model.predict_proba(X_val)[:, 1]
                test_proba = model.predict_proba(X_test)[:, 1]
            else:
                # Fallback: decision_function
                val_proba = model.decision_function(X_val)
                test_proba = model.decision_function(X_test)

            oof_preds[name][val_idx] = val_proba
            test_preds_folds[name].append(test_proba)

        print()

    # Average test predictions across folds
    test_preds = {}
    for name in model_factories.keys():
        test_stack = np.vstack(test_preds_folds[name])  # (n_folds, n_test)
        test_preds[name] = test_stack.mean(axis=0)

    # OOF performance per model
    print("\nOOF validation performance per base model:")
    for name in model_factories.keys():
        preds = (oof_preds[name] >= 0.5).astype(int)
        acc = accuracy_score(y, preds)
        print(f"  {name}: {acc:.5f}")

    # =====================================================
    # Ensemble weight optimization using random search
    # =====================================================

    print("\nOptimizing ensemble weights (random Dirichlet search)...")

    model_names = list(model_factories.keys())
    n_models = len(model_names)

    oof_matrix = np.vstack([oof_preds[m] for m in model_names])    # (n_models, n_samples)
    test_matrix = np.vstack([test_preds[m] for m in model_names])  # (n_models, n_test)

    best_acc = 0.0
    best_w = None
    n_iter = 2000  # you can increase to 5000+ if it's fast enough on your machine

    rng = np.random.default_rng(42)

    for i in range(n_iter):
        # Sample weights that sum to 1
        w = rng.dirichlet(alpha=np.ones(n_models))
        blended = np.dot(w, oof_matrix)
        preds = (blended >= 0.5).astype(int)
        acc = accuracy_score(y, preds)

        if acc > best_acc:
            best_acc = acc
            best_w = w

        if (i + 1) % 200 == 0:
            print(f"  Iteration {i+1}/{n_iter} | Best ensemble acc so far: {best_acc:.5f}")

    print("\nBest ensemble OOF accuracy:", best_acc)
    print("Best weights:")
    for name, weight in zip(model_names, best_w):
        print(f"  {name}: {weight:.4f}")

    # Final test blending with best weights
    final_test_scores = np.dot(best_w, test_matrix)
    final_preds = np.where(final_test_scores >= 0.5, "Stayed", "Left")

    submission = pd.DataFrame({
        id_col: test[id_col],
        target_col: final_preds
    })

    out_name = "submission_simple_ensemble_v3.csv"
    submission.to_csv(out_name, index=False)
    print(f"\n✅ Saved: {out_name}")


if __name__ == "__main__":
    main()
