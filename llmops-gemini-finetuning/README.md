# From MLOps to LLMOps: Fine-Tuning Gemini on IRIS

## Objective

This assignment applies LLMOps principles to IRIS classification using Vertex AI
supervised fine-tuning of Gemini.

Two representations are compared:

- V1 — Raw feature values
- V2 — Natural language description

## Dataset

IRIS dataset with 150 samples and three classes:

- setosa
- versicolor
- virginica

Train samples: 120  
Test samples: 30  
Random state: 42

## Models

Base model:

gemini-2.5-flash

Region:

us-central1

Hyperparameters:

- Epochs: 3
- Learning rate multiplier: 1.0

## V1 — Raw Representation

Example input:

sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2

Expected output:

setosa

## V2 — Natural Language Representation

Example input:

A flower specimen has a sepal length of 5.1 cm, sepal width of 3.5 cm,
petal length of 1.4 cm, and petal width of 0.2 cm. Identify the iris species.

Expected output:

This is Iris setosa.

## Vertex AI Tuning Jobs

V1:

3972410312858009600

V2:

1427876523393679360

Both jobs use identical hyperparameters. The data representation is the
experimental variable.

## Evaluation Metrics

The tuned models were evaluated on held-out test data using:

- Accuracy
- Per-class precision
- Per-class recall
- Format compliance rate

Format compliance requires an exact valid species name:

setosa
versicolor
virginica

## Results

Evaluated on the 30 held-out test samples per representation.

| Metric | V1 Raw | V2 Description |
|---|---:|---:|
| Accuracy | 56.67% | 36.67% |
| Setosa Precision | 52.63% | 35.71% |
| Versicolor Precision | 42.86% | 0.00% |
| Virginica Precision | 100.00% | 50.00% |
| Setosa Recall | 100.00% | 100.00% |
| Versicolor Recall | 30.00% | 0.00% |
| Virginica Recall | 40.00% | 10.00% |
| Format Compliance | 0.00% | 0.00% |

Full details: `results/RESULTS.md` and `results/final_metrics.json`.

## LLMOps Practices

- Dataset versioning
- Prompt/data representation versioning
- GCS storage
- Managed fine-tuning
- Experiment tracking
- Held-out evaluation
- Output validation
- Git-based workflow

## Conclusion

The raw feature representation (V1) performed better than the natural-language
representation (V2): 56.67% versus 36.67% accuracy. Both fine-tuned models scored
0% on strict format compliance because they often returned explanatory text instead
of a single species name. For small tabular classification, a general LLM is a poor
fit compared with a simple classical model, and LLM evaluation must include
output-format compliance as well as accuracy.
