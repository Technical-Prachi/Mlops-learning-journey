import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp

TRAIN_PATH = "data/iris.csv"
OUTPUT_DIR = "explainability-fairness-drift/results"
PRODUCTION_PATH = "explainability-fairness-drift/data/iris_production.csv"

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load original training data
train_df = pd.read_csv(TRAIN_PATH)

# Create simulated production data
production_df = train_df.copy()

# Introduce distribution drift
production_df["petal_length"] = (
    production_df["petal_length"] + 0.8
)

production_df["petal_width"] = (
    production_df["petal_width"] + 0.3
)

# Save simulated production dataset
production_df.to_csv(PRODUCTION_PATH, index=False)

print("Simulated production dataset created.")
print(f"Saved to: {PRODUCTION_PATH}")

# KS test for distribution drift
results = []

for feature in FEATURES:

    train_values = train_df[feature]
    production_values = production_df[feature]

    statistic, p_value = ks_2samp(
        train_values,
        production_values
    )

    drift_detected = p_value < 0.05

    results.append({
        "feature": feature,
        "ks_statistic": statistic,
        "p_value": p_value,
        "drift_detected": drift_detected
    })

    print(f"\nFeature: {feature}")
    print(f"KS statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Drift detected: {drift_detected}")

results_df = pd.DataFrame(results)

results_path = os.path.join(
    OUTPUT_DIR,
    "drift_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(f"\nDrift results saved to: {results_path}")

# Distribution plots
for feature in FEATURES:

    plt.figure(figsize=(8, 5))

    plt.hist(
        train_df[feature],
        bins=15,
        alpha=0.6,
        label="Training"
    )

    plt.hist(
        production_df[feature],
        bins=15,
        alpha=0.6,
        label="Production"
    )

    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Training vs Production: {feature}")
    plt.legend()

    plot_path = os.path.join(
        OUTPUT_DIR,
        f"drift_{feature}.png"
    )

    plt.tight_layout()
    plt.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {plot_path}")

print("\nData drift analysis completed successfully.")
