import json
from pathlib import Path

DATA_DIR = Path("llmops-gemini-finetuning/data")

def convert(src, dst):
    with open(src) as f:
        records = [json.loads(line) for line in f]

    with open(dst, "w") as f:
        for r in records:
            obj = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": r["input_text"]}
                        ]
                    },
                    {
                        "role": "model",
                        "parts": [
                            {"text": r["output_text"]}
                        ]
                    }
                ]
            }
            f.write(json.dumps(obj) + "\n")

files = [
    ("iris_v1_train.jsonl", "iris_v1_train_gemini.jsonl"),
    ("iris_v1_test.jsonl", "iris_v1_test_gemini.jsonl"),
    ("iris_v2_train.jsonl", "iris_v2_train_gemini.jsonl"),
    ("iris_v2_test.jsonl", "iris_v2_test_gemini.jsonl"),
]

for src, dst in files:
    convert(DATA_DIR / src, DATA_DIR / dst)
    print("Created:", dst)

