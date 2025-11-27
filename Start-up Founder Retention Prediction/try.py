import pandas as pd

# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Quick checks
print(train.shape)
print(train.head())
print(test.head())

# Check missing values
print(train.isnull().sum())

# Target variable distribution
print(train['retention_status'].value_counts())
