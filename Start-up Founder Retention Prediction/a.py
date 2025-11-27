# a.py
"""
Push-to-760 pipeline (NO NN)
- Median impute + log1p for numerics
- Frequency encoding for cats
- KFold smoothed target encoding (OOF) for cats + interactions
- OHE for low-cardinality cats
- Interaction creation done safely on train_cat/test_cat
- OOF training of XGBoost and LightGBM
- OOF meta training (Ridge)
- Coarse + refined weight sweep for final blend
Outputs: submission_push_to_760.csv
"""

import warnings
warnings.filterwarnings("ignore")
import gc
import numpy as np
import pandas as pd
from math import log1p
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb

# -------------------------
# Config
# -------------------------
SEED = 42
NFOLDS = 5
np.random.seed(SEED)

# -------------------------
# Column definitions — keep same as your dataset
# -------------------------
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

id_col = "founder_id"
target_col = "retention_status"

# -------------------------
# Helper functions
# -------------------------
def smooth_target_encoding(series, target, min_samples_leaf=100, smoothing=10):
    """
    Compute smoothed target mean per category.
    Returns mapping dict.
    """
    stats = pd.concat([series, target], axis=1).groupby(series.name)[target.name].agg(['mean','count'])
    prior = target.mean()
    # smoothing function
    smoothing_val = 1/(1 + np.exp(-(stats['count'] - min_samples_leaf)/smoothing))
    stats['smooth'] = prior * (1 - smoothing_val) + stats['mean'] * smoothing_val
    return stats['smooth'].to_dict()

def kfold_target_encoding(series_train_df, series_test_df, col_name, target_series, n_splits=NFOLDS, seed=SEED, min_samples_leaf=100, smoothing=10):
    """
    Out-of-fold target encoding for a single column.
    series_train_df and series_test_df are pandas Series (train_cat[col], test_cat[col]) or DataFrames column-like
    Returns (oof_series, test_array)
    """
    oof = pd.Series(index=series_train_df.index, dtype=float)
    test_vals = np.zeros(len(series_test_df))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for _, (tr_idx, val_idx) in enumerate(skf.split(series_train_df, target_series)):
        tr_cat = series_train_df.iloc[tr_idx]
        tr_y = target_series.iloc[tr_idx]
        mapping = smooth_target_encoding(tr_cat, tr_y, min_samples_leaf=min_samples_leaf, smoothing=smoothing)
        val_cat = series_train_df.iloc[val_idx]
        oof.iloc[val_idx] = val_cat.map(mapping).fillna(target_series.mean())
        # apply mapping to test and accumulate
        test_cat = series_test_df
        test_vals += test_cat.map(mapping).fillna(target_series.mean()).values / n_splits
    return oof.fillna(target_series.mean()), test_vals

def freq_encoding(series):
    counts = series.value_counts(dropna=False)
    return series.map(counts).astype(float)

# -------------------------
# Load data
# -------------------------
print("Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = train[target_col].map({"Stayed": 1, "Left": 0})

test_ids = test[id_col].copy()

# -------------------------
# Basic preprocessing: numeric impute + log1p
# -------------------------
print("Numeric imputation and log1p transform...")
num_imputer = SimpleImputer(strategy='median')
train_num = pd.DataFrame(num_imputer.fit_transform(train[numerical_cols]), columns=numerical_cols, index=train.index)
test_num = pd.DataFrame(num_imputer.transform(test[numerical_cols]), columns=numerical_cols, index=test.index)

# log1p transform (clip negatives to zero)
for c in numerical_cols:
    train_num[c] = np.log1p(np.clip(train_num[c], a_min=0, a_max=None))
    test_num[c] = np.log1p(np.clip(test_num[c], a_min=0, a_max=None))

# categorical fill (work on separate copies)
train_cat = train[categorical_cols].fillna("NA").copy()
test_cat = test[categorical_cols].fillna("NA").copy()

# -------------------------
# Frequency encoding for categorical columns
# -------------------------
print("Applying frequency encoding...")
for c in categorical_cols:
    train[c + "_freq"] = freq_encoding(train_cat[c])
    test[c + "_freq"] = freq_encoding(test_cat[c])
freq_cols = [c + "_freq" for c in categorical_cols]

# -------------------------
# Create interaction features safely on train_cat/test_cat
# -------------------------
print("Creating interaction features safely...")
cardinality = {c: train_cat[c].nunique() for c in categorical_cols}
top_cats = sorted(cardinality.items(), key=lambda x: x[1], reverse=True)[:4]
top_cat_names = [x[0] for x in top_cats]

interactions = []
for i in range(len(top_cat_names)):
    for j in range(i+1, len(top_cat_names)):
        colname = f"{top_cat_names[i]}__{top_cat_names[j]}"
        interactions.append(colname)
        train_cat[colname] = train_cat[top_cat_names[i]].astype(str) + "___" + train_cat[top_cat_names[j]].astype(str)
        test_cat[colname] = test_cat[top_cat_names[i]].astype(str) + "___" + test_cat[top_cat_names[j]].astype(str)

# -------------------------
# Out-of-fold target encoding for categorical columns and interactions
# -------------------------
print("Applying KFold target encoding (OOF) for categories and interactions...")
oof_target_enc = pd.DataFrame(index=train.index)
test_target_enc = pd.DataFrame(index=test.index)

# encode original categorical columns
for c in categorical_cols:
    oof_te, test_te = kfold_target_encoding(train_cat[c], test_cat[c], c, y, n_splits=NFOLDS, seed=SEED, min_samples_leaf=100, smoothing=20)
    oof_target_enc[c + "_te"] = oof_te
    test_target_enc[c + "_te"] = test_te

# encode interaction columns
for c in interactions:
    oof_te, test_te = kfold_target_encoding(train_cat[c], test_cat[c], c, y, n_splits=NFOLDS, seed=SEED, min_samples_leaf=200, smoothing=30)
    oof_target_enc[c + "_te"] = oof_te
    test_target_enc[c + "_te"] = test_te

# -------------------------
# One-Hot encode a few low-cardinality columns (<=6)
# -------------------------
low_card_cols = [c for c in categorical_cols if train_cat[c].nunique() <= 6]
print("Low-cardinality columns to OHE:", low_card_cols)

if len(low_card_cols) > 0:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    combined = pd.concat([train_cat[low_card_cols], test_cat[low_card_cols]], axis=0)
    ohe.fit(combined)
    ohe_train = pd.DataFrame(ohe.transform(train_cat[low_card_cols]), index=train.index,
                             columns=[f"ohe_{i}" for i in range(ohe.transform(train_cat[low_card_cols]).shape[1])])
    ohe_test = pd.DataFrame(ohe.transform(test_cat[low_card_cols]), index=test.index,
                            columns=[f"ohe_{i}" for i in range(ohe.transform(test_cat[low_card_cols]).shape[1])])
else:
    ohe_train = pd.DataFrame(index=train.index)
    ohe_test = pd.DataFrame(index=test.index)

# -------------------------
# Final feature assembly
# -------------------------
print("Assembling final feature frames...")
X_features = pd.concat([train_num, oof_target_enc, train[freq_cols], ohe_train], axis=1)
X_test_features = pd.concat([test_num, test_target_enc, test[freq_cols], ohe_test], axis=1)

# safety fill
X_features = X_features.fillna(X_features.mean())
X_test_features = X_test_features.fillna(X_features.mean().reindex(X_test_features.columns).fillna(0))

print("Final shapes -> train:", X_features.shape, "test:", X_test_features.shape)

# -------------------------
# OOF training utility
# -------------------------
def oof_train_predict_estimator(estimator_fn, X, y, X_test, n_splits=NFOLDS, seed=SEED, params=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_preds = np.zeros((X_test.shape[0], n_splits))
    models = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
        model = estimator_fn() if params is None else estimator_fn(**params)
        model.fit(X_tr, y_tr)
        if hasattr(model, "predict_proba"):
            oof[val_idx] = model.predict_proba(X_val_fold)[:,1]
            test_preds[:, fold] = model.predict_proba(X_test)[:,1]
        else:
            try:
                oof[val_idx] = model.decision_function(X_val_fold)
                test_preds[:, fold] = model.decision_function(X_test)
            except Exception:
                oof[val_idx] = model.predict(X_val_fold)
                test_preds[:, fold] = model.predict(X_test)
        models.append(model)
        print(f" Fold {fold+1} done")
    test_pred_mean = test_preds.mean(axis=1)
    oof_acc = accuracy_score(y, (oof >= 0.5).astype(int))
    return oof, test_pred_mean, models, oof_acc

# -------------------------
# Train XGBoost OOF
# -------------------------
print("\nOOF training XGBoost...")
def build_xgb():
    return xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1
    )

Xf = X_features.reset_index(drop=True)
Xtf = X_test_features.reset_index(drop=True)
y_series = y.reset_index(drop=True)

oof_xgb, test_xgb, xgb_models, acc_xgb = oof_train_predict_estimator(build_xgb, Xf, y_series, Xtf)
print("XGB OOF acc:", acc_xgb)

# -------------------------
# Train LightGBM OOF
# -------------------------
print("\nOOF training LightGBM...")
def build_lgb():
    return lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=60,
        colsample_bytree=0.8,
        subsample=0.8,
        reg_lambda=2.0,
        random_state=SEED,
        n_jobs=-1
    )

oof_lgb, test_lgb, lgb_models, acc_lgb = oof_train_predict_estimator(build_lgb, Xf, y_series, Xtf)
print("LGB OOF acc:", acc_lgb)

# -------------------------
# Meta-model training (Ridge) on OOF preds
# -------------------------
print("\nTraining Ridge meta-model on OOF predictions...")
meta_train = pd.DataFrame({'xgb': oof_xgb, 'lgb': oof_lgb})
meta_test = pd.DataFrame({'xgb': test_xgb, 'lgb': test_lgb})

meta = RidgeClassifierCV(alphas=np.logspace(-3,1,10))
meta.fit(meta_train, y_series)
meta_oof = meta.predict(meta_train)
meta_acc = accuracy_score(y_series, meta_oof)
print("Meta OOF acc:", meta_acc)

# -------------------------
# Weight sweep (coarse then refine)
# -------------------------
print("\nSearching best weight combination (coarse grid)...")
best = (0,0,-1.0)  # w_xgb, w_lgb, score
steps = np.arange(0.0,1.01,0.05)
for wx in steps:
    for wl in steps:
        if wx + wl > 1.0: 
            continue
        blend_oof = wx * oof_xgb + wl * oof_lgb
        sc = accuracy_score(y_series, (blend_oof >= 0.5).astype(int))
        if sc > best[2]:
            best = (wx, wl, sc)
print("Coarse best:", best)

# refine search around best
wx0, wl0, sc0 = best
best2 = (0,0,-1.0)
for wx in np.arange(max(0,wx0-0.1), min(1,wx0+0.1)+1e-9, 0.01):
    for wl in np.arange(max(0,wl0-0.1), min(1,wl0+0.1)+1e-9, 0.01):
        if wx + wl > 1: continue
        blend_oof = wx * oof_xgb + wl * oof_lgb
        sc = accuracy_score(y_series, (blend_oof >= 0.5).astype(int))
        if sc > best2[2]:
            best2 = (wx, wl, sc)
print("Refined best:", best2)

# Final weights
wx, wl, _ = best2
print("Final blend weights (xgb, lgb):", wx, wl)

final_test_preds = wx * test_xgb + wl * test_lgb
final_labels = ["Stayed" if p >= 0.5 else "Left" for p in final_test_preds]

submission = pd.DataFrame({id_col: test_ids, target_col: final_labels})
submission.to_csv("submission_push_to_760.csv", index=False)
print("Saved submission_push_to_760.csv")

# Print OOF metrics
print("\nOOF metrics:")
print("XGB OOF acc:", acc_xgb)
print("LGB OOF acc:", acc_lgb)
print("Meta OOF acc:", meta_acc)
print("Blend OOF acc (refined):", best2[2])

# cleanup
del Xf, Xtf, X_features, X_test_features
gc.collect()
