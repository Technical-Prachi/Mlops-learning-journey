import json
import re
import csv
from pathlib import Path

from guarded_pipeline import guarded_predict


SPECIES = {"setosa", "versicolor", "virginica"}


def load_finetune_data(path, version):
    data = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            prompt = obj["input_text"]
            expected = obj["output_text"].strip().lower()

            matches = re.findall(
                r"\b(setosa|versicolor|virginica)\b",
                expected
            )

            if len(set(matches)) == 1:
                expected = matches[0]

            data.append((prompt, expected))

    return data


def run_attack_tests(filename):
    with open(filename, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = 0
    blocked = 0

    for row in rows:
        prompt = row["input_prompt"]
        version = row["model_version"]

        result = guarded_predict(
            prompt,
            version
        )

        total += 1

        if result.get("blocked"):
            blocked += 1

    rate = blocked / total if total else 0

    return {
        "total": total,
        "blocked": blocked,
        "block_rate": rate
    }


def evaluate_clean(version, path):

    data = load_finetune_data(path, version)

    correct = 0
    input_blocked = 0
    output_blocked = 0
    model_errors = 0
    total = len(data)

    predictions = []

    for i, (prompt, expected) in enumerate(data, 1):

        result = guarded_predict(
            prompt,
            version
        )

        stage = result.get("stage")
        reason = result.get("reason")

        predicted = None

        if stage == "model" and reason == "model_error":

            model_errors += 1

        elif stage == "input_guardrail":

            input_blocked += 1

        elif stage == "output_guardrail":

            output_blocked += 1

        elif stage == "completed":

            predicted = (
                result.get("response", "")
                .lower()
                .strip()
            )

            if predicted == expected:
                correct += 1

        predictions.append({
            "index": i,
            "expected": expected,
            "predicted": predicted or "",
            "blocked": result.get("blocked"),
            "stage": stage,
            "reason": reason
        })

        print(
            f"{version.upper()} {i:02d}/{total} | "
            f"expected={expected} | "
            f"predicted={predicted} | "
            f"stage={stage} | "
            f"reason={reason}"
        )

    accuracy = correct / total if total else 0

    false_positive_rate = (
        input_blocked / total
        if total
        else 0
    )

    return {
        "total": total,
        "correct": correct,
        "input_blocked": input_blocked,
        "output_blocked": output_blocked,
        "model_errors": model_errors,
        "accuracy": accuracy,
        "false_positive_rate": false_positive_rate,
        "predictions": predictions
    }


def main():

    Path("llm-guardrails-red-teaming/results").mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n========================================")
    print("TASK 5 — GUARDRail EFFECTIVENESS")
    print("========================================")

    print("\n[1] Injection block rate")

    injection = run_attack_tests(
        "llm-guardrails-red-teaming/results/injection_results.csv"
    )

    print(
        f"Blocked: {injection['blocked']}/"
        f"{injection['total']}"
    )

    print(
        f"Injection block rate: "
        f"{injection['block_rate'] * 100:.2f}%"
    )

    print("\n[2] Leakage block rate")

    leakage = run_attack_tests(
        "llm-guardrails-red-teaming/results/leakage_results.csv"
    )

    print(
        f"Blocked: {leakage['blocked']}/"
        f"{leakage['total']}"
    )

    print(
        f"Leakage block rate: "
        f"{leakage['block_rate'] * 100:.2f}%"
    )

    print("\n[3] Clean V1 evaluation")

    v1 = evaluate_clean(
        "v1",
        "llmops-gemini-finetuning/data/iris_v1_test.jsonl"
    )

    print("\n[4] Clean V2 evaluation")

    v2 = evaluate_clean(
        "v2",
        "llmops-gemini-finetuning/data/iris_v2_test.jsonl"
    )

    baseline = {}

    try:
        with open(
            "llmops-gemini-finetuning/results/final_metrics.json",
            encoding="utf-8"
        ) as f:

            old = json.load(f)

        baseline["V1"] = old["V1"]["accuracy"]
        baseline["V2"] = old["V2"]["accuracy"]

    except Exception:

        baseline["V1"] = None
        baseline["V2"] = None

    v1_delta = (
        v1["accuracy"] - baseline["V1"]
        if baseline["V1"] is not None
        else None
    )

    v2_delta = (
        v2["accuracy"] - baseline["V2"]
        if baseline["V2"] is not None
        else None
    )

    summary = {
        "injection": injection,
        "leakage": leakage,
        "V1": {
            "baseline_accuracy": baseline["V1"],
            "guarded_accuracy": v1["accuracy"],
            "accuracy_delta": v1_delta,
            "false_positive_rate": v1["false_positive_rate"],
            "input_blocked": v1["input_blocked"],
            "output_blocked": v1["output_blocked"],
            "model_errors": v1["model_errors"]
        },
        "V2": {
            "baseline_accuracy": baseline["V2"],
            "guarded_accuracy": v2["accuracy"],
            "accuracy_delta": v2_delta,
            "false_positive_rate": v2["false_positive_rate"],
            "input_blocked": v2["input_blocked"],
            "output_blocked": v2["output_blocked"],
            "model_errors": v2["model_errors"]
        }
    }

    with open(
        "llm-guardrails-red-teaming/results/guardrail_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    with open(
        "llm-guardrails-red-teaming/results/guardrail_summary.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(["Metric", "Value"])

        writer.writerow([
            "Injection Block Rate",
            f"{injection['block_rate'] * 100:.2f}%"
        ])

        writer.writerow([
            "Leakage Block Rate",
            f"{leakage['block_rate'] * 100:.2f}%"
        ])

        writer.writerow([
            "V1 False Positive Rate",
            f"{v1['false_positive_rate'] * 100:.2f}%"
        ])

        writer.writerow([
            "V2 False Positive Rate",
            f"{v2['false_positive_rate'] * 100:.2f}%"
        ])

        writer.writerow([
            "V1 Baseline Accuracy",
            baseline["V1"]
        ])

        writer.writerow([
            "V1 Guarded Accuracy",
            v1["accuracy"]
        ])

        writer.writerow([
            "V1 Accuracy Delta",
            v1_delta
        ])

        writer.writerow([
            "V1 Input Blocks",
            v1["input_blocked"]
        ])

        writer.writerow([
            "V1 Output Blocks",
            v1["output_blocked"]
        ])

        writer.writerow([
            "V1 Model Errors",
            v1["model_errors"]
        ])

        writer.writerow([
            "V2 Baseline Accuracy",
            baseline["V2"]
        ])

        writer.writerow([
            "V2 Guarded Accuracy",
            v2["accuracy"]
        ])

        writer.writerow([
            "V2 Accuracy Delta",
            v2_delta
        ])

        writer.writerow([
            "V2 Input Blocks",
            v2["input_blocked"]
        ])

        writer.writerow([
            "V2 Output Blocks",
            v2["output_blocked"]
        ])

        writer.writerow([
            "V2 Model Errors",
            v2["model_errors"]
        ])

    print("\n========================================")
    print("FINAL GUARDRail METRICS")
    print("========================================")

    print(
        f"Injection Block Rate : "
        f"{injection['block_rate'] * 100:.2f}%"
    )

    print(
        f"Leakage Block Rate   : "
        f"{leakage['block_rate'] * 100:.2f}%"
    )

    print(
        f"V1 False Positive    : "
        f"{v1['false_positive_rate'] * 100:.2f}%"
    )

    print(
        f"V2 False Positive    : "
        f"{v2['false_positive_rate'] * 100:.2f}%"
    )

    print(
        f"V1 Guarded Accuracy  : "
        f"{v1['accuracy'] * 100:.2f}%"
    )

    print(
        f"V2 Guarded Accuracy  : "
        f"{v2['accuracy'] * 100:.2f}%"
    )

    if v1_delta is not None:
        print(
            f"V1 Accuracy Delta    : "
            f"{v1_delta * 100:+.2f}%"
        )

    if v2_delta is not None:
        print(
            f"V2 Accuracy Delta    : "
            f"{v2_delta * 100:+.2f}%"
        )

    print(
        f"V1 Input Blocks      : "
        f"{v1['input_blocked']}"
    )

    print(
        f"V1 Output Blocks     : "
        f"{v1['output_blocked']}"
    )

    print(
        f"V1 Model Errors      : "
        f"{v1['model_errors']}"
    )

    print(
        f"V2 Input Blocks      : "
        f"{v2['input_blocked']}"
    )

    print(
        f"V2 Output Blocks     : "
        f"{v2['output_blocked']}"
    )

    print(
        f"V2 Model Errors      : "
        f"{v2['model_errors']}"
    )

    print("\nResults saved in llm-guardrails-red-teaming/results/")


if __name__ == "__main__":
    main()
