import json
import re
import csv
from pathlib import Path
from google import genai

PROJECT = "iris-mlops-500704"
LOCATION = "us-central1"

V1_ENDPOINT = (
    "projects/885601899302/locations/us-central1/"
    "endpoints/8209891146638295040"
)

V2_ENDPOINT = (
    "projects/885601899302/locations/us-central1/"
    "endpoints/3828310921671868416"
)

client = genai.Client(
    vertexai=True,
    project=PROJECT,
    location=LOCATION
)

SPECIES = {"setosa", "versicolor", "virginica"}

def load_test(path):
    rows = []

    with open(path) as f:
        for line in f:
            obj = json.loads(line)

            text = obj["contents"][0]["parts"][0]["text"]
            expected = obj["contents"][1]["parts"][0]["text"]

            # V2 expected: "This is Iris setosa."
            expected_species = expected.lower()

            for s in SPECIES:
                if s in expected_species:
                    expected_species = s
                    break

            rows.append((text, expected_species))

    return rows


def predict(endpoint, prompt):
    response = client.models.generate_content(
        model=endpoint,
        contents=prompt
    )

    return (response.text or "").strip()


def evaluate(name, endpoint, data):
    predictions = []

    for i, (prompt, expected) in enumerate(data, 1):

        try:
            raw = predict(endpoint, prompt)
        except Exception as e:
            raw = f"ERROR: {e}"

        normalized = raw.lower().strip()

        # Strict format compliance:
        # response must be exactly setosa/versicolor/virginica
        format_ok = normalized in SPECIES

        # For semantic classification accuracy, detect species
        predicted_species = None

        for species in SPECIES:
            if re.search(r"\b" + species + r"\b", normalized):
                predicted_species = species
                break

        correct = predicted_species == expected

        predictions.append({
            "index": i,
            "expected": expected,
            "raw_response": raw,
            "predicted": predicted_species or "",
            "correct": correct,
            "format_compliant": format_ok
        })

        print(
            f"{name} {i:02d}/30 | "
            f"expected={expected} | "
            f"predicted={predicted_species} | "
            f"correct={correct} | "
            f"format={format_ok}"
        )

    accuracy = sum(x["correct"] for x in predictions) / len(predictions)

    format_rate = (
        sum(x["format_compliant"] for x in predictions)
        / len(predictions)
    )

    metrics = {
        "model": name,
        "accuracy": accuracy,
        "format_compliance": format_rate,
        "total": len(predictions)
    }

    return predictions, metrics


def precision_recall(predictions):
    results = {}

    for species in SPECIES:

        tp = sum(
            1 for x in predictions
            if x["expected"] == species
            and x["predicted"] == species
        )

        fp = sum(
            1 for x in predictions
            if x["expected"] != species
            and x["predicted"] == species
        )

        fn = sum(
            1 for x in predictions
            if x["expected"] == species
            and x["predicted"] != species
        )

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0

        results[species] = {
            "precision": precision,
            "recall": recall
        }

    return results


v1_data = load_test("llmops-gemini-finetuning/data/iris_v1_test_gemini.jsonl")
v2_data = load_test("llmops-gemini-finetuning/data/iris_v2_test_gemini.jsonl")

print("\n========== V1 EVALUATION ==========\n")
v1_predictions, v1_metrics = evaluate(
    "V1",
    V1_ENDPOINT,
    v1_data
)

print("\n========== V2 EVALUATION ==========\n")
v2_predictions, v2_metrics = evaluate(
    "V2",
    V2_ENDPOINT,
    v2_data
)

v1_pr = precision_recall(v1_predictions)
v2_pr = precision_recall(v2_predictions)

print("\n========== FINAL RESULTS ==========\n")

for name, metrics, pr in [
    ("V1", v1_metrics, v1_pr),
    ("V2", v2_metrics, v2_pr)
]:

    print(name)
    print("Accuracy:", round(metrics["accuracy"] * 100, 2), "%")
    print(
        "Format compliance:",
        round(metrics["format_compliance"] * 100, 2),
        "%"
    )

    for species in SPECIES:
        print(
            species,
            "precision=",
            round(pr[species]["precision"], 4),
            "recall=",
            round(pr[species]["recall"], 4)
        )

# Save detailed predictions
Path("llmops-gemini-finetuning/results").mkdir(exist_ok=True)

with open("llmops-gemini-finetuning/results/v1_predictions.json", "w") as f:
    json.dump(v1_predictions, f, indent=2)

with open("llmops-gemini-finetuning/results/v2_predictions.json", "w") as f:
    json.dump(v2_predictions, f, indent=2)

final = {
    "V1": {
        **v1_metrics,
        "per_class": v1_pr
    },
    "V2": {
        **v2_metrics,
        "per_class": v2_pr
    }
}

with open("llmops-gemini-finetuning/results/final_metrics.json", "w") as f:
    json.dump(final, f, indent=2)

print("\nResults saved in llmops-gemini-finetuning/results/")
