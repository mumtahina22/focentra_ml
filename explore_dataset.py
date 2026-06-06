import pandas as pd

df = pd.read_csv("data/raw/student_productivity.csv")

print("=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== FIRST 3 ROWS ===")
print(df.head(3).to_string())

print("\n=== NULL COUNTS ===")
print(df.isnull().sum())

print("\n=== NUMERIC STATS ===")
print(df.describe().to_string())