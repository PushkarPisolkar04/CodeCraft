import pandas as pd

# Load test data
test_data = pd.read_csv('data/test.csv')

# Print column names
print("Column names:")
print(test_data.columns.tolist())

# Print first few rows
print("\nFirst few rows:")
print(test_data.head())

print("\nDataset Info:")
print("-" * 50)
print(test_data.info()) 