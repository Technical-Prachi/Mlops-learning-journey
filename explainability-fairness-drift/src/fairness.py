import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame

DATA_PATH = "explainability-fairness-drift/data/iris_with_location.csv"
OUTPUT_PATH = "explainability-fairness-drift/results/fairness_metrics.csv"

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df["species"]
sensitive = df["location"]

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X,
    y,
    sensitive,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

metrics = {
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "recall": lambda y_true, y_pred: recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
}

metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=s_test,
)

print("\n===== OVERALL METRICS =====")
print(metric_frame.overall)

print("\n===== METRICS BY LOCATION =====")
print(metric_frame.by_group)

print("\n===== PERFORMANCE GAP =====")
print(metric_frame.by_group.max() - metric_frame.by_group.min())

result = metric_frame.by_group.reset_index()
result.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved fairness results to: {OUTPUT_PATH}")
