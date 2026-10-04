import os
import joblib
import pandas as pd
from datetime import datetime
from google.cloud import storage
import mlflow
import mlflow.sklearn
from feast import FeatureStore

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------
# Configuration
# -------------------------
BUCKET_NAME = "mlops-course-iris-mlops-500704"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris_Classification")
# -------------------------
# Load Features from Feast
# -------------------------
print("Loading features from Feast...")

store = FeatureStore(
    repo_path="feast_feature_store/feature_repo"
)

entity_df = pd.DataFrame({
    "id": list(range(150)),
    "event_timestamp": [datetime.now()] * 150
})

df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
        "iris_features:target",
    ],
).to_df()

print("Features fetched successfully from Feast.")

# -------------------------
# Prepare Training Data
# -------------------------
X = df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
]

y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# -------------------------
# Train Model
# -------------------------
best_accuracy = 0
best_model = None

n_estimators_list = [50, 100]
max_depth_list = [3, 5]

for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:

        with mlflow.start_run():

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
            )

            model.fit(X_train, y_train)

            prediction = model.predict(X_test)

            accuracy = accuracy_score(y_test, prediction)

            print(
                f"n_estimators={n_estimators}, "
                f"max_depth={max_depth}, "
                f"accuracy={accuracy:.4f}"
            )

            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            mlflow.log_metric("accuracy", accuracy)

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="iris-classifier",
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

print("Best Accuracy:", best_accuracy)

model = best_model

# -------------------------
# Save Model
# -------------------------
os.makedirs("artifacts", exist_ok=True)



# -------------------------
# Upload Artifacts to GCS
# -------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

with open("artifacts/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}")

metrics_blob = bucket.blob(f"outputs/{timestamp}/metrics.txt")
metrics_blob.upload_from_filename("artifacts/metrics.txt")

print("Artifacts uploaded successfully.")
print("Timestamp:", timestamp)
