import pandas as pd
import numpy as np

INPUT_PATH = "data/iris.csv"
OUTPUT_PATH = "explainability-fairness-drift/data/iris_with_location.csv"

RANDOM_SEED = 42

df = pd.read_csv(INPUT_PATH)

rng = np.random.default_rng(RANDOM_SEED)
df["location"] = rng.integers(0, 2, size=len(df))

df.to_csv(OUTPUT_PATH, index=False)

print("Dataset created successfully.")
print(f"Shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())

print("\nLocation distribution:")
print(df["location"].value_counts().sort_index())

print("\nFirst 5 rows:")
print(df.head())
