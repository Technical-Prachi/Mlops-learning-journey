import mlflow
import mlflow.sklearn
import pandas as pd
from feast import FeatureStore

# -------------------------
# MLflow Tracking
# -------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Apna best Run ID
RUN_ID = "e2fbff425ad249a7a376b3d01ee9d36b"

print("Loading model from MLflow Run...")

model = mlflow.sklearn.load_model(
    f"runs:/{RUN_ID}/model"
)

print("Model loaded successfully!")

# -------------------------
# Load Features from Feast
# -------------------------

store = FeatureStore(
    repo_path="feast_feature_store/feature_repo"
)

features = store.get_online_features(
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ],
    entity_rows=[
        {"id": 0}
    ],
).to_dict()

print("Features fetched from Feast:")
print(features)

sample = pd.DataFrame({
    "sepal_length": features["sepal_length"],
    "sepal_width": features["sepal_width"],
    "petal_length": features["petal_length"],
    "petal_width": features["petal_width"],
})

prediction = model.predict(sample)

print("Prediction:", prediction[0])
