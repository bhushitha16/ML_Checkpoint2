import numpy as np
import pandas as pd

# Load average probabilities (use the simple model predictions you want to boost)
probs = np.load("probs_seed_42.npy")  # OR use your favourite seed model

# Class order (same as before)
classes = ['Cluster_A','Cluster_B','Cluster_C','Cluster_D','Cluster_E']

# Boost factors based on under/over prediction
boost = {
    'Cluster_A': 1.20,
    'Cluster_B': 1.10,
    'Cluster_C': 1.05,
    'Cluster_D': 1.00,
    'Cluster_E': 0.95
}

# Convert dict to vector in class order
boost_vec = np.array([boost[c] for c in classes])

# Apply boosting
probs = probs * boost_vec

# Renormalize
probs = probs / probs.sum(axis=1, keepdims=True)

# Final labels
final_labels = probs.argmax(axis=1)
final_classes = [classes[i] for i in final_labels]

# Save submission
test = pd.read_csv("test.csv")
sub = pd.DataFrame({
    "participant_id": test["participant_id"],
    "personality_cluster": final_classes
})
sub.to_csv("submission_boosted.csv", index=False)
print("Saved submission_boosted.csv")
