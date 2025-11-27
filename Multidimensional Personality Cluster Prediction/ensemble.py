import numpy as np
import pandas as pd

# All seeds you ran
seeds = [42, 123, 2024, 777, 9999]

# Load all probabilities
prob_list = [np.load(f"probs_seed_{s}.npy") for s in seeds]

# Average predictions
avg_probs = sum(prob_list) / len(prob_list)

# Mapping classes back
classes = ['Cluster_A', 'Cluster_B', 'Cluster_C', 'Cluster_D', 'Cluster_E']

# Load test file
test = pd.read_csv("test.csv")

# Select final labels
final_labels = avg_probs.argmax(axis=1)
final_classes = [classes[i] for i in final_labels]

# Create submission
sub = pd.DataFrame({
    "participant_id": test["participant_id"],
    "personality_cluster": final_classes
})

sub.to_csv("submission_ensemble.csv", index=False)
print("Saved: submission_ensemble.csv")
