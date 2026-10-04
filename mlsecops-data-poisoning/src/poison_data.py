import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path("data/iris.csv")
OUTPUT_DIR = Path("mlsecops-data-poisoning/data")

SEED = 42

def create_poisoned_dataset(df, corruption_rate, seed=42):
    rng = np.random.default_rng(seed)

    poisoned = df.copy()

    feature_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]

    label_column = "species"

    n_samples = len(poisoned)
    n_poison = int(round(n_samples * corruption_rate))

    poison_indices = rng.choice(
        n_samples,
        size=n_poison,
        replace=False
    )

    # Random feature values
    for col in feature_columns:
        poisoned.loc[poison_indices, col] = rng.uniform(
            df[col].min(),
            df[col].max(),
            size=n_poison
        )

    # Random labels
    labels = df[label_column].unique()
    poisoned.loc[poison_indices, label_column] = rng.choice(
        labels,
        size=n_poison,
        replace=True
    )

    return poisoned, poison_indices


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    print("Original dataset shape:", df.shape)
    print("Columns:", list(df.columns))

    # Clean baseline
    clean_file = OUTPUT_DIR / "iris_clean.csv"
    df.to_csv(clean_file, index=False)

    print(f"Clean dataset saved: {clean_file}")

    corruption_levels = {
        "5": 0.05,
        "10": 0.10,
        "50": 0.50
    }

    for level, rate in corruption_levels.items():
        poisoned_df, indices = create_poisoned_dataset(
            df,
            rate,
            seed=SEED
        )

        output_file = OUTPUT_DIR / f"iris_poisoned_{level}.csv"
        poisoned_df.to_csv(output_file, index=False)

        print(
            f"{level}% poisoning: "
            f"{len(indices)} samples corrupted -> {output_file}"
        )


if __name__ == "__main__":
    main()
