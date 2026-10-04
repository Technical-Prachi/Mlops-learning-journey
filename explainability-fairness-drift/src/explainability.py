import os
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA_PATH = "explainability-fairness-drift/data/iris_with_location.csv"
OUTPUT_DIR = "explainability-fairness-drift/results"

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully.")
print("Classes:", model.classes_)

# SHAP full-dataset explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

print("SHAP values generated.")
print("SHAP shape:", shap_values.values.shape)

# Generate one summary plot for each class
for class_index, class_name in enumerate(model.classes_):

    plt.figure()

    shap.summary_plot(
        shap_values[:, :, class_index],
        X,
        show=False
    )

    plt.title(f"SHAP Summary Plot - {class_name}")

    output_path = os.path.join(
        OUTPUT_DIR,
        f"shap_{class_name}.png"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

print("\nAll SHAP plots generated successfully.")
