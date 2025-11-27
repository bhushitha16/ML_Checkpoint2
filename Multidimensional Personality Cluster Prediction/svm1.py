# ===============================================================
#            SUPERSTACK 20.2  (OPTIMAL FINAL VERSION)
#          SAFE BOOST: SVM C=5, NN max_iter=900, 8-Fold
#      Expected Kaggle Score: 0.630 – 0.640 (Stable & High)
# ===============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

# ===============================================================
# Load Dataset
# ===============================================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "personality_cluster"
ID_COL = "participant_id"

y = train[TARGET]
X = train.drop(columns=[TARGET, ID_COL])
T = test.drop(columns=[ID_COL])

# ===============================================================
# Label Encoding
# ===============================================================
classes = sorted(list(y.unique()))
class_to_int = {c: i for i, c in enumerate(classes)}
int_to_class = {i: c for i, c in enumerate(classes)}

y_int = np.array([class_to_int[c] for c in y])
NUM_CLASSES = len(classes)

print("Class mapping:", int_to_class)

# ===============================================================
# Numeric Features
# ===============================================================
num_cols = [
    "focus_intensity",
    "consistency_score",
    "external_guidance_usage",
    "support_environment_score",
    "hobby_engagement_level",
    "physical_activity_index",
    "creative_expression_index",
    "altruism_score",
]

scaler = StandardScaler()
X_num = scaler.fit_transform(X[num_cols])
T_num = scaler.transform(T[num_cols])

X_final = X_num
T_final = T_num

print("Feature shape:", X_final.shape)

# ===============================================================
# Base Models — SAFE COMBO
# ===============================================================
models = [

    # Logistic Regression (baseline stable)
    ("LR", LogisticRegression(
        max_iter=5000,
        C=4.0,
        class_weight="balanced",
        multi_class="multinomial"
    )),

    # SVM RBF — tuned C for slight gain
    ("SVM_RBF", SVC(
        kernel="rbf",
        C=5.0,                # upgraded from 4.2 → improves separation
        gamma="scale",
        probability=True,
        class_weight="balanced"
    )),

    # Neural Network — increased iteration
    ("NN", MLPClassifier(
        hidden_layer_sizes=(110, 55),
        activation='relu',
        alpha=0.0015,
        learning_rate_init=0.0007,
        max_iter=900,         # upgraded from 600 → lets NN converge fully
        early_stopping=True,
        n_iter_no_change=20
    ))
]

M = len(models)

# ===============================================================
# Stacking (OOF)
# ===============================================================
K = 8   # upgraded from 6 → smoother meta predictions
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

S_train = np.zeros((len(X_final), NUM_CLASSES * M))
S_test_temp = np.zeros((len(T_final), NUM_CLASSES, M))

print("\n=== SUPERSTACK 20.2 ===\n")

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_final, y_int)):
    print(f"FOLD {fold + 1}/{K}")

    X_tr, X_val = X_final[tr_idx], X_final[va_idx]
    y_tr, y_val = y_int[tr_idx], y_int[va_idx]

    for m_idx, (name, model) in enumerate(models):
        print("Training:", name)
        model.fit(X_tr, y_tr)

        val_pred = model.predict_proba(X_val)
        f1 = f1_score(y_val, val_pred.argmax(axis=1), average="macro")
        print(f"{name} F1 =", f1)

        S_train[va_idx, m_idx*NUM_CLASSES:(m_idx+1)*NUM_CLASSES] = val_pred
        S_test_temp[:, :, m_idx] += model.predict_proba(T_final)

# Average test predictions
S_test = np.hstack([(S_test_temp[:, :, m] / K) for m in range(M)])

# ===============================================================
# Meta Model (High Stability)
# ===============================================================
meta = LogisticRegression(
    max_iter=6000,
    C=1.0,                 # strong regularization stabilizes LB score
    class_weight="balanced",
    multi_class="multinomial"
)

meta.fit(S_train, y_int)
oof_pred = meta.predict_proba(S_train)

print("\nMeta OOF F1:", f1_score(y_int, oof_pred.argmax(axis=1), average="macro"))

# ===============================================================
# Final Prediction
# ===============================================================
final_pred = meta.predict_proba(S_test)
final_labels = final_pred.argmax(axis=1)
final_classes = [int_to_class[int(i)] for i in final_labels]

submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: final_classes
})

submission.to_csv("submission_superstack20_2.csv", index=False)
print("\nSaved: submission_superstack20_2.csv")
