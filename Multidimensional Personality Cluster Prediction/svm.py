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

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "personality_cluster"
ID = "participant_id"

y = train[TARGET]
X = train.drop(columns=[TARGET])
T = test.copy()

classes = sorted(list(y.unique()))
class_to_int = {c: i for i, c in enumerate(classes)}
int_to_class = {i: c for i, c in enumerate(classes)}
y_int = np.array([class_to_int[c] for c in y])

# -----------------------------
# Basic numeric-only scaling
# -----------------------------
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

# -----------------------------
# Base Models: LR, SVM, NN
# -----------------------------
models = [
    ("LR", LogisticRegression(max_iter=3000, class_weight="balanced", multi_class="multinomial")),
    ("SVM_RBF", SVC(kernel="rbf", C=3.0, gamma="scale", probability=True, class_weight="balanced")),
    ("NN", MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                         alpha=0.001, learning_rate_init=0.001,
                         max_iter=500, early_stopping=True))
]

M = len(models)
NUM_CLASSES = len(classes)

# -----------------------------
# Stacking: OOF
# -----------------------------
S_train = np.zeros((len(X_final), M * NUM_CLASSES))
S_test_temp = np.zeros((len(T_final), NUM_CLASSES, M))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_final, y_int)):
    print(f"FOLD {fold+1}")

    X_tr, X_val = X_final[tr_idx], X_final[va_idx]
    y_tr, y_val = y_int[tr_idx], y_int[va_idx]

    for m, (name, model) in enumerate(models):
        print("Training:", name)
        model.fit(X_tr, y_tr)
        val_pred = model.predict_proba(X_val)
        f1 = f1_score(y_val, val_pred.argmax(axis=1), average="macro")
        print(name, "F1 =", f1)

        S_train[va_idx, m*NUM_CLASSES:(m+1)*NUM_CLASSES] = val_pred
        S_test_temp[:, :, m] += model.predict_proba(T_final)

S_test = np.hstack([S_test_temp[:, :, m] / 5 for m in range(M)])

# -----------------------------
# Meta Model (LR)
# -----------------------------
meta = LogisticRegression(max_iter=5000, class_weight="balanced", multi_class="multinomial")
meta.fit(S_train, y_int)

oof_meta = meta.predict_proba(S_train)
print("\nMeta OOF F1 =", f1_score(y_int, oof_meta.argmax(axis=1), average="macro"))

# -----------------------------
# Final Prediction
# -----------------------------
final_pred = meta.predict_proba(S_test)
final_labels = final_pred.argmax(axis=1)

final_classes = [int_to_class[int(i)] for i in final_labels]

sub = pd.DataFrame({
    "participant_id": T[ID],
    "personality_cluster": final_classes
})

sub.to_csv("submission_svm_nn_lr.csv", index=False)
print("\nSaved submission_svm_nn_lr.csv")
