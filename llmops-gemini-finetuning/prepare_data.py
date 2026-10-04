import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load IRIS
iris = load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

# Same split for both v1 and v2
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

def write_jsonl(path, X_data, y_data, version):
    with open(path, "w") as f:
        for features, label in zip(X_data, y_data):

            sl, sw, pl, pw = features
            species = target_names[label]

            if version == "v1":
                record = {
                    "input_text": (
                        f"sepal_length: {sl:.1f}, "
                        f"sepal_width: {sw:.1f}, "
                        f"petal_length: {pl:.1f}, "
                        f"petal_width: {pw:.1f}"
                    ),
                    "output_text": species
                }

            else:
                record = {
                    "input_text": (
                        f"A flower specimen has a sepal length of {sl:.1f} cm, "
                        f"sepal width of {sw:.1f} cm, "
                        f"petal length of {pl:.1f} cm, "
                        f"and petal width of {pw:.1f} cm. "
                        f"Identify the iris species."
                    ),
                    "output_text": f"This is Iris {species}."
                }

            f.write(json.dumps(record) + "\n")


# v1 - Raw
write_jsonl(
    "llmops-gemini-finetuning/data/iris_v1_train.jsonl",
    X_train,
    y_train,
    "v1"
)

write_jsonl(
    "llmops-gemini-finetuning/data/iris_v1_test.jsonl",
    X_test,
    y_test,
    "v1"
)

# v2 - Natural Language
write_jsonl(
    "llmops-gemini-finetuning/data/iris_v2_train.jsonl",
    X_train,
    y_train,
    "v2"
)

write_jsonl(
    "llmops-gemini-finetuning/data/iris_v2_test.jsonl",
    X_test,
    y_test,
    "v2"
)

print("Dataset preparation complete!")
print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
