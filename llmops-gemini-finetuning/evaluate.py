import json
import re
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

SPECIES = ["setosa", "versicolor", "virginica"]


def normalize_prediction(text):
    return text.strip()


def format_compliant(text):
    return text.strip() in SPECIES


def extract_species(text):
    text = text.strip().lower()

    if text in SPECIES:
        return text

    return None


def evaluate(predictions, expected):
    compliant = [format_compliant(p) for p in predictions]

    valid_preds = []
    valid_expected = []

    for p, e in zip(predictions, expected):
        species = extract_species(p)
        if species is not None:
            valid_preds.append(species)
            valid_expected.append(e)

    accuracy = accuracy_score(
        expected,
        [
            extract_species(p) if extract_species(p) is not None else "__invalid__"
            for p in predictions
        ]
    )

    precision, recall, _, _ = precision_recall_fscore_support(
        valid_expected,
        valid_preds,
        labels=SPECIES,
        zero_division=0
    ) if valid_preds else ([0, 0, 0], [0, 0, 0], None, None)

    compliance_rate = sum(compliant) / len(compliant) * 100

    return {
        "accuracy": accuracy * 100,
        "format_compliance_rate": compliance_rate,
        "precision": dict(zip(SPECIES, precision)),
        "recall": dict(zip(SPECIES, recall)),
    }


def load_expected(path, version):
    expected = []

    with open(path) as f:
        for line in f:
            r = json.loads(line)

            if version == "v1":
                expected.append(r["output_text"].strip().lower())
            else:
                # "This is Iris setosa." -> setosa
                match = re.search(
                    r"(setosa|versicolor|virginica)",
                    r["output_text"].lower()
                )
                expected.append(match.group(1))

    return expected


if __name__ == "__main__":

    # Placeholder predictions.
    # These will be replaced by actual Vertex AI tuned-model predictions.
    print("Evaluation framework ready.")
    print("V1 test samples:",
          len(load_expected("llmops-gemini-finetuning/data/iris_v1_test.jsonl", "v1")))
    print("V2 test samples:",
          len(load_expected("llmops-gemini-finetuning/data/iris_v2_test.jsonl", "v2")))

