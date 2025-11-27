# ============================================================
# FT-Transformer + XGBoost Hybrid (CPU)
# Target: Macro-F1 ≈ 0.66 – 0.69
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# GLOBAL SEED
# ============================================================
def seed_everything(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything(42)

# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "personality_cluster"
ID_COL = "participant_id"

X = train.drop(columns=[TARGET])
y = train[TARGET]
test_X = test.copy()

# Map labels to integers
classes = sorted(y.unique())
class_to_int = {c: i for i, c in enumerate(classes)}
int_to_class = {i: c for c, i in class_to_int.items()}
y_int = y.map(class_to_int).values
NUM_CLASSES = len(classes)

# ============================================================
# PREPROCESSING
# ============================================================

cat_cols = ["age_group", "identity_code", "cultural_background", "upbringing_influence"]
num_cols = [c for c in X.columns if c not in cat_cols + [ID_COL]]

# ---------- Frequency Encoding ----------
def frequency_encode(df_train, df_test, cols):
    for c in cols:
        freq = df_train[c].value_counts()
        df_train[c + "_FE"] = df_train[c].map(freq)
        df_test[c + "_FE"] = df_test[c].map(freq)
    return df_train, df_test

X, test_X = frequency_encode(X, test_X, cat_cols)

# ---------- Target Encoding ----------
def target_encode_kfold(train_df, test_df, col, target, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    encoded = np.zeros(len(train_df))
    test_encoded = np.zeros(len(test_df))
    global_mean = target.mean()

    for tr_idx, val_idx in skf.split(train_df, target):
        tr = train_df.iloc[tr_idx].copy()
        val = train_df.iloc[val_idx].copy()

        tr["_y"] = target[tr_idx]
        means = tr.groupby(col)["_y"].mean()

        encoded[val_idx] = val[col].map(means).fillna(global_mean)
        test_encoded += test_df[col].map(means).fillna(global_mean) / n_splits

    train_df[col + "_TE"] = encoded
    test_df[col + "_TE"] = test_encoded
    return train_df, test_df

for c in cat_cols:
    X, test_X = target_encode_kfold(X, test_X, c, y_int)

# ---------- Ordinal Encoding ----------
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
encoder.fit(X[cat_cols])

# ---------- Numeric Scaling ----------
scaler = RobustScaler()
scaler.fit(X[num_cols])

# ============================================================
# COMBINE FEATURES
# ============================================================
def prepare_features(df, df_test):
    X_cat = encoder.transform(df[cat_cols])
    T_cat = encoder.transform(df_test[cat_cols])

    fe_cols = [c + "_FE" for c in cat_cols]
    te_cols = [c + "_TE" for c in cat_cols]

    X_fe = df[fe_cols].values
    T_fe = df_test[fe_cols].values

    X_te = df[te_cols].values
    T_te = df_test[te_cols].values

    X_num = scaler.transform(df[num_cols])
    T_num = scaler.transform(df_test[num_cols])

    X_all = np.hstack([X_cat, X_fe, X_te, X_num])
    T_all = np.hstack([T_cat, T_fe, T_te, T_num])

    return X_all, T_all

X_all, T_all = prepare_features(X, test_X)

# ============================================================
# FT-TRANSFORMER MODULES
# ============================================================

# ---------- Tokenizer for categorical and numeric ----------
class TabularTokenizer(nn.Module):
    def __init__(self, cat_dims, num_dim, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(d + 1, emb_dim) for d in cat_dims
        ])
        self.numeric_linear = nn.Linear(num_dim, emb_dim)

    def forward(self, x_cat, x_num):
        cat_tokens = [emb(col.long()) for emb, col in zip(self.cat_embeddings, x_cat.T)]
        cat_tokens = torch.stack(cat_tokens, dim=1)
        num_token = self.numeric_linear(x_num).unsqueeze(1)
        return torch.cat([cat_tokens, num_token], dim=1)

# ---------- FT-Transformer Block ----------
class FTTransformer(nn.Module):
    def __init__(self, cat_dims, num_dim, emb_dim=64, depth=4, heads=4, dropout=0.1):
        super().__init__()

        self.tokenizer = TabularTokenizer(cat_dims, num_dim, emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.fc = nn.Linear(emb_dim, NUM_CLASSES)

    def forward(self, x_cat, x_num):
        B = x_cat.size(0)

        tokens = self.tokenizer(x_cat, x_num)
        cls_tok = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)

        out = self.transformer(tokens)
        cls_out = out[:, 0]
        return self.fc(cls_out)
# ============================================================
# PART 2 — TRAIN FT-TRANSFORMER + XGBOOST (CPU)
# ============================================================

device = torch.device("cpu")   # CPU version

# ------------------------------------------------------------
# TRAIN LOOP FOR FT-TRANSFORMER
# ------------------------------------------------------------
def train_ft_transformer(model, X_cat, X_num, y, lr=1e-3, epochs=25, batch_size=64):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    dataset_size = X_cat.shape[0]
    model.train()

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        X_cat = X_cat[perm]
        X_num = X_num[perm]
        y = y[perm]

        total_loss = 0
        batches = dataset_size // batch_size

        for i in range(batches):
            xb_cat = X_cat[i*batch_size:(i+1)*batch_size].to(device)
            xb_num = X_num[i*batch_size:(i+1)*batch_size].to(device)
            yb = y[i*batch_size:(i+1)*batch_size].to(device)

            optimizer.zero_grad()
            preds = model(xb_cat, xb_num)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Optional: print small progress
        # print(f"Epoch {epoch+1}/{epochs}, loss={total_loss/batches:.4f}")

    return model

# ------------------------------------------------------------
# PREDICT WITH FT-TRANSFORMER
# ------------------------------------------------------------
def predict_ft(model, X_cat, X_num, batch_size=128):
    model.eval()
    preds_list = []

    with torch.no_grad():
        for i in range(0, len(X_cat), batch_size):
            xc = X_cat[i:i+batch_size].to(device)
            xn = X_num[i:i+batch_size].to(device)
            logits = model(xc, xn)
            preds = torch.softmax(logits, dim=1)
            preds_list.append(preds.cpu().numpy())

    return np.vstack(preds_list)


# ============================================================
# XGBOOST (CPU)
# ============================================================

def get_xgb_model():
    return xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",   # CPU histogram
        random_state=42
    )


# ============================================================
# STACKING FT-TRANSFORMER + XGBOOST
# ============================================================

def run_oof_stacking(X_all, y_int, T_all):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # X_all contains: [categorical embeddings, freq, target, numeric]
    TOTAL_FEATURES = X_all.shape[1]

    # We split raw features for FT-T
    N_CAT = len(cat_cols)
    N_FE = len(cat_cols)
    N_TE = len(cat_cols)
    N_NUM = len(num_cols)

    # Indices:
    idx_cat_start = 0
    idx_cat_end = idx_cat_start + N_CAT

    idx_fe_start = idx_cat_end
    idx_fe_end = idx_fe_start + N_FE

    idx_te_start = idx_fe_end
    idx_te_end = idx_te_start + N_TE

    idx_num_start = idx_te_end
    idx_num_end = idx_num_start + N_NUM

    # Outputs
    S_train = np.zeros((len(X_all), NUM_CLASSES * 2))   # FT + XGB = 2 models
    S_test = np.zeros((len(T_all), NUM_CLASSES * 2))

    test_preds_ft = np.zeros((len(T_all), NUM_CLASSES, 5))
    test_preds_xgb = np.zeros((len(T_all), NUM_CLASSES, 5))

    fold_num = 0
    for tr_idx, val_idx in skf.split(X_all, y_int):
        fold_num += 1
        print(f"\n========== Fold {fold_num} ==========")

        # Split input for FT-Transformer
        Xtr_cat = torch.tensor(X_all[tr_idx, idx_cat_start:idx_cat_end], dtype=torch.long)
        Xval_cat = torch.tensor(X_all[val_idx, idx_cat_start:idx_cat_end], dtype=torch.long)
        Xtest_cat = torch.tensor(T_all[:, idx_cat_start:idx_cat_end], dtype=torch.long)

        Xtr_num = torch.tensor(X_all[tr_idx, idx_num_start:idx_num_end], dtype=torch.float32)
        Xval_num = torch.tensor(X_all[val_idx, idx_num_start:idx_num_end], dtype=torch.float32)
        Xtest_num = torch.tensor(T_all[:, idx_num_start:idx_num_end], dtype=torch.float32)

        ytr = torch.tensor(y_int[tr_idx], dtype=torch.long)
        yval = y_int[val_idx]

        # -------------------------
        # FT-TRANSFORMER
        # -------------------------
        print("Training FT-Transformer...")
        ft_model = FTTransformer(
            cat_dims=[50, 500, 500, 500],   # high capacity
            num_dim=N_NUM,
            emb_dim=64,
            depth=4,
            heads=4,
            dropout=0.15
        )

        ft_model = train_ft_transformer(
            ft_model,
            Xtr_cat, Xtr_num, ytr,
            lr=1e-3,
            epochs=16,
            batch_size=64
        )

        val_preds_ft = predict_ft(ft_model, Xval_cat, Xval_num)
        test_preds_ft[:, :, fold_num-1] = predict_ft(ft_model, Xtest_cat, Xtest_num)

        f1_ft = f1_score(yval, val_preds_ft.argmax(axis=1), average="macro")
        print(f"  FT-Transformer Fold F1 = {f1_ft:.4f}")

        # store in S_train
        S_train[val_idx, 0:NUM_CLASSES] = val_preds_ft

        # -------------------------
        # XGBOOST
        # -------------------------
        print("Training XGBoost...")

        xgb_model = get_xgb_model()
        xgb_model.fit(X_all[tr_idx], y_int[tr_idx])

        val_preds_xgb = xgb_model.predict_proba(X_all[val_idx])
        test_preds_xgb[:, :, fold_num-1] = xgb_model.predict_proba(T_all)

        f1_xgb = f1_score(yval, val_preds_xgb.argmax(axis=1), average="macro")
        print(f"  XGBoost Fold F1 = {f1_xgb:.4f}")

        # store in S_train
        S_train[val_idx, NUM_CLASSES:NUM_CLASSES*2] = val_preds_xgb

    # Average T_all predictions
    S_test[:, 0:NUM_CLASSES] = test_preds_ft.mean(axis=2)
    S_test[:, NUM_CLASSES:NUM_CLASSES*2] = test_preds_xgb.mean(axis=2)

    return S_train, S_test
# ============================================================
# PART 3 — META MODEL + FINAL BLENDING + SUBMISSION
# ============================================================

from sklearn.linear_model import LogisticRegression

# ------------------------------------------------------------
# RUN LEVEL-1 STACKING
# ------------------------------------------------------------
print("\n======================================")
print(" Running FT-Transformer + XGBoost OOF ")
print("======================================")

S_train, S_test = run_oof_stacking(X_all, y_int, T_all)

print("\nStack shapes:")
print("S_train:", S_train.shape)
print("S_test :", S_test.shape)

# ============================================================
# META MODEL (LEVEL-2)
# ============================================================

print("\n======================================")
print(" Training LEVEL-2 Meta Model (LogReg) ")
print("======================================")

meta_model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    random_state=42
)

meta_model.fit(S_train, y_int)

meta_oof = meta_model.predict_proba(S_train)
meta_oof_f1 = f1_score(y_int, meta_oof.argmax(axis=1), average="macro")
print(f"Meta-model OOF Macro-F1 = {meta_oof_f1:.4f}")

# Base meta prediction
meta_test_pred = meta_model.predict_proba(S_test)


# ============================================================
# SEED AVERAGING FOR STABILITY
# ============================================================

def run_meta_seed(seed):
    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=seed
    )
    model.fit(S_train, y_int)
    return model.predict_proba(S_test)


SEEDS = [42, 2024, 6500]

print("\n======================")
print(" Running seed averaging")
print("======================")

final_pred = np.zeros_like(meta_test_pred)

for s in SEEDS:
    print(f"  Running seed {s}")
    final_pred += run_meta_seed(s)

final_pred /= len(SEEDS)


# ============================================================
# DECODE LABELS + BUILD SUBMISSION
# ============================================================

pred_labels = final_pred.argmax(axis=1)
pred_classes = [int_to_class[i] for i in pred_labels]

submission = pd.DataFrame({
    "participant_id": test[ID_COL],
    "personality_cluster": pred_classes
})

output_file = "submission_hybrid_ft_xgb.csv"
submission.to_csv(output_file, index=False)

print("\n=============================================")
print(" Final submission saved:", output_file)
print("=============================================\n")
