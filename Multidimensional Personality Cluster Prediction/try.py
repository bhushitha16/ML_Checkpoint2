import pandas as pd

train = pd.read_csv("train.csv")

print("Unique classes:", train["personality_cluster"].nunique())
print("Value counts:")
print(train["personality_cluster"].value_counts())
print(sorted(y.unique()))
print(class_to_int)
print(int_to_class)

