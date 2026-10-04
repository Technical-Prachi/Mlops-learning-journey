import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_DIR = Path("mlsecops-data-poisoning/data")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris_MLSecOps_Poisoning")


DATASETS = {
    "0%": DATA_DIR / "iris_clean.csv",
    "5%": DATA_DIR / "iris_poisoned_5.csv",
    "10%": DATA_DIR / "iris_poisoned_10.csv",
    "50%": DATA_DIR / "iris_poisoned_50.csv",
}


def train_experiment(poisoning_level, file_path):

    print("\n" + "=" * 60)
    print(f"Training: {poisoning_level}")
    print(f"Dataset: {file_path}")
    print("=" * 60)

    df = pd.read_csv(file_path)

    feature_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    X = df[feature_columns]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    with mlflow.start_run(run_name=f"iris_poisoning_{poisoning_level}"):

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        # Log poisoning information
        mlflow.log_param("poisoning_level", poisoning_level)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        return {
            "poisoning_level": poisoning_level,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def main():

    results = []

    for poisoning_level, file_path in DATASETS.items():
        result = train_experiment(
            poisoning_level,
            file_path,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    output_file = Path("mlsecops-data-poisoning/results/metrics.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
