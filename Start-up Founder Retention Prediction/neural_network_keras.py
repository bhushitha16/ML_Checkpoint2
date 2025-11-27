import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Keras/TensorFlow for NN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature columns
numerical_cols = [
    'founder_age', 'years_with_startup', 'monthly_revenue_generated', 'funding_rounds_led',
    'distance_from_investor_hub', 'num_dependents', 'years_since_founding'
]
categorical_cols = [
    'founder_gender', 'founder_role', 'work_life_balance_rating', 'venture_satisfaction',
    'startup_performance_rating', 'working_overtime', 'education_background', 'personal_status',
    'startup_stage', 'team_size_category', 'remote_operations', 'leadership_scope',
    'innovation_support', 'startup_reputation', 'founder_visibility'
]

id_col = 'founder_id'
target_col = 'retention_status'

X = train[numerical_cols + categorical_cols]
y = train[target_col].map({'Stayed': 1, 'Left': 0})
X_test = test[numerical_cols + categorical_cols]

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Fit and transform training data
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

print("Processed feature shape:", X_processed.shape)

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
print("Train class distribution:", np.bincount(y_train))
print("Validation class distribution:", np.bincount(y_val))

# Build Neural Network Model (minimal regularization)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(48, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop],
    verbose=2
)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation accuracy: {val_acc:.4f}")

# Predict and prepare submission
test_pred_probs = model.predict(X_test_processed, verbose=0).flatten()
test_preds_labels = ['Stayed' if p >= 0.5 else 'Left' for p in test_pred_probs]

submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: test_preds_labels
})
submission.to_csv('submission_nn.csv', index=False)
print("Neural Network submission file saved as submission_nn.csv")
