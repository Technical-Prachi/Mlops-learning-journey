# MLSecOps: Data Poisoning Experiment on the IRIS Pipeline

## Overview

This assignment explores Machine Learning Security Operations (MLSecOps) by identifying major ML security threat vectors and simulating a data poisoning attack on the IRIS classification pipeline.

The experiment creates clean and poisoned versions of the IRIS dataset with 5%, 10%, and 50% corruption levels. A Random Forest classifier is trained on each dataset, and MLflow is used to track and compare accuracy, precision, recall, and F1 score.

---

## Objectives

The objectives of this assignment are:

1. Identify major security threat vectors in ML systems.
2. Simulate data poisoning attacks at different severity levels.
3. Measure the effect of poisoned training data on model performance.
4. Track experiments using MLflow.
5. Analyze the relationship between data poisoning, data quality, and model performance.
6. Discuss production-level MLSecOps mitigation strategies.

---

# Task 1 - ML Security Threat Vectors

## 1. Data Poisoning

### Target Stage
Data ingestion and training.

### Description
Data poisoning occurs when an attacker intentionally modifies or injects malicious or incorrect samples into the training dataset.

The poisoned samples can change the learned decision boundaries and reduce model accuracy or cause targeted misclassification.

### Example
An attacker modifies a portion of training records before the dataset reaches the ML training pipeline. The trained model then learns from incorrect feature values and labels.

### Mitigation
- Data validation
- Statistical profiling
- Anomaly detection
- Data provenance tracking
- Schema validation
- Dataset versioning
- Quality gates before training

---

## 2. Adversarial Examples

### Target Stage
Inference.

### Description
Adversarial examples are inputs that have been deliberately modified to cause a trained model to produce an incorrect prediction.

Unlike data poisoning, adversarial examples target the model after training.

### Example
A small perturbation is applied to an image so that a classification model incorrectly predicts its class even though the modified image appears almost unchanged to a human.

### Mitigation
- Adversarial testing
- Input validation
- Robust model training
- Anomaly detection
- Monitoring unusual inference inputs

---

## 3. Model Extraction

### Target Stage
Model serving and inference API.

### Description
Model extraction occurs when an attacker repeatedly queries a deployed model and uses the outputs to approximate or reproduce its behavior.

### Example
An attacker sends thousands of carefully selected queries to a prediction API and collects the responses to train a substitute model.

### Mitigation
- Rate limiting
- Authentication
- Query monitoring
- Access controls
- Logging
- Limiting prediction details returned by APIs

---

## 4. Prompt Injection

### Target Stage
LLM inference and application layer.

### Description
Prompt injection occurs when malicious instructions are included in user input to manipulate an LLM into ignoring its intended instructions or revealing information it should not provide.

### Example
A malicious user provides instructions designed to override the application's system instructions and make an LLM reveal sensitive information.

### Mitigation
- Input validation
- Prompt boundary enforcement
- Output validation
- Tool access restrictions
- Least-privilege design
- Monitoring and logging

---

# Task 2 - IRIS Data Poisoning

The original IRIS dataset was used as the clean baseline.

The poisoning process replaces all four numerical features of selected samples with randomly generated values and assigns a randomly generated class label.

The four affected features are:

- sepal_length
- sepal_width
- petal_length
- petal_width

## Dataset Variants

| Dataset | Corruption Level | Corrupted Samples |
|---|---:|---:|
| iris_clean.csv | 0% | 0 |
| iris_poisoned_5.csv | 5% | 8 |
| iris_poisoned_10.csv | 10% | 16 |
| iris_poisoned_50.csv | 50% | 80 |

The poisoning was implemented in:

`mlsecops-data-poisoning/src/poison_data.py`

Generated datasets are stored under:

`mlsecops-data-poisoning/data/`

---

# Task 3 - MLflow Experiments

A Random Forest classifier was trained independently on all four dataset variants.

The following metrics were tracked using MLflow:

- Accuracy
- Precision
- Recall
- F1 Score

The MLflow experiment is:

`Iris_MLSecOps_Poisoning`

The training implementation is available in:

`mlsecops-data-poisoning/src/train_mlsecops.py`

The experiment contains four runs:

- iris_poisoning_0%
- iris_poisoning_5%
- iris_poisoning_10%
- iris_poisoning_50%

---

# Task 4 - Validation and Results

## MLflow Results

| Poisoning Level | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| 0% | 0.9688 | 0.9716 | 0.9688 | 0.9687 |
| 5% | 0.9688 | 0.9716 | 0.9688 | 0.9687 |
| 10% | 0.9688 | 0.9714 | 0.9688 | 0.9685 |
| 50% | 0.6562 | 0.6577 | 0.6562 | 0.6541 |

The detailed results are stored in:

`mlsecops-data-poisoning/results/metrics.csv`

## Analysis

### At which poisoning level does degradation become noticeable?

In this experiment, significant degradation becomes visible at the 50% poisoning level.

The accuracy decreases from:

96.88% → 65.63%

This represents a drop of approximately 31.25 percentage points.

At 5% and 10% poisoning, the model performance remains almost unchanged.

### Which metrics are affected first?

At 10% poisoning, small changes are visible in precision and F1 score.

However, the changes are very small:

- Precision: 0.9716 → 0.9714
- F1: 0.9687 → 0.9685

The major degradation occurs at 50% poisoning, where accuracy, precision, recall, and F1 all decrease substantially.

### Does the model still learn meaningful patterns at 50% corruption?

Yes, the model does not become completely random.

At 50% corruption, accuracy is approximately 65.63%, which indicates that the model still captures some meaningful patterns from the remaining clean information.

However, the substantial performance reduction demonstrates that severe data poisoning can significantly damage model reliability.

---

# Task 5 - MLSecOps Mitigation Strategies

## 1. Statistical Validation

Training data should be profiled before model training.

Useful checks include:

- Feature ranges
- Mean and standard deviation
- Distribution changes
- Missing values
- Duplicate records
- Unexpected categorical values

Large deviations from expected distributions can indicate possible poisoning.

---

## 2. Anomaly Detection

Anomaly detection can identify samples that differ significantly from normal training data.

Possible approaches include:

- Isolation Forest
- Local Outlier Factor
- Clustering
- Statistical outlier detection
- Distance-based detection

Suspicious samples can be reviewed or removed before training.

---

## 3. Data Provenance Tracking

The pipeline should record:

- Data source
- Dataset version
- Collection timestamp
- Data transformation history
- Data owner
- Dataset checksum/hash

DVC and other dataset versioning mechanisms can help identify unexpected changes to training data.

---

## 4. Schema Enforcement

A data quality gate should verify:

- Expected columns exist
- Correct data types are used
- Feature ranges are valid
- Labels belong to the allowed classes
- Missing values are handled
- Unexpected records are rejected

Training should not proceed if critical validation checks fail.

---

## 5. Quality Gates

Automated validation should run before training.

Example pipeline:

```text
Data Ingestion
      |
      v
Schema Validation
      |
      v
Statistical Validation
      |
      v
Anomaly Detection
      |
      v
Data Quality Gate
      |
      v
Training
      |
      v
MLflow Evaluation
      |
      v
Model Deployment

If the data fails the quality gate, training should be blocked until the dataset is investigated.

Data Quantity vs Data Quality

Increasing the amount of data does not automatically solve a data quality problem.

If additional data contains poisoned samples, increasing the dataset size may also increase the amount of malicious information available to the model.

For example, suppose a dataset contains 20% poisoned samples.

Increasing the dataset size while maintaining the same poisoning ratio does not improve the clean-data ratio. The model is still learning from contaminated data.

The important factor is therefore not only the total number of samples, but also the ratio of clean to corrupted samples.

Clean Data Ratio

The clean data ratio can be represented as:

Clean Data Ratio = Clean Samples / Total Samples

For the experiments in this assignment:

Poisoning	Approximate Clean Ratio
0%	100%
5%	95%
10%	90%
50%	50%

At 50% poisoning, half of the training samples are corrupted. This substantially reduces the amount of reliable information available to the model.

Production Implication

If data quality is compromised, collecting more data is useful only when the additional data is trustworthy and improves the clean-data ratio.

Therefore:

More data is not necessarily better than better data.

A production ML pipeline should prioritize:

Trusted data sources
Data validation
Provenance
Anomaly detection
Dataset versioning
Quality thresholds
Continuous monitoring

The minimum dataset size required for reliable training depends on model complexity, class balance, feature quality, and the proportion of trustworthy samples.

Project Structure
mlsecops-data-poisoning/
│
├── data/
│   ├── iris_clean.csv
│   ├── iris_poisoned_5.csv
│   ├── iris_poisoned_10.csv
│   └── iris_poisoned_50.csv
│
├── results/
│   └── metrics.csv
│
├── screenshots/
│
└── src/
    ├── poison_data.py
    └── train_mlsecops.py
Reproduction
Generate Poisoned Datasets

From the repository root:

python mlsecops-data-poisoning/src/poison_data.py

This generates:

mlsecops-data-poisoning/data/iris_clean.csv
mlsecops-data-poisoning/data/iris_poisoned_5.csv
mlsecops-data-poisoning/data/iris_poisoned_10.csv
mlsecops-data-poisoning/data/iris_poisoned_50.csv
Run MLsecOps Experiments

Start the MLflow server:

mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --allowed-hosts '*' \
  --cors-allowed-origins 'https://524453c485b0cb8f-dot-us-central1.notebooks.googleusercontent.com'

Then run:

python mlsecops-data-poisoning/src/train_mlsecops.py

The experiment results are tracked under:

Iris_MLSecOps_Poisoning
Conclusion

The experiment demonstrates that the IRIS classification model is relatively robust to low levels of random data poisoning in this particular setup, with little measurable degradation at 5% and 10% corruption.

However, at 50% poisoning, model performance drops substantially from approximately 96.88% accuracy to 65.63%.

This demonstrates the importance of integrating security controls into the ML lifecycle.

MLSecOps should therefore treat training data as a security-sensitive asset and implement validation, provenance tracking, anomaly detection, quality gates, and continuous monitoring before models are trained and deployed.

Assignment Artifacts
Poisoning script: mlsecops-data-poisoning/src/poison_data.py
Training and MLflow script: mlsecops-data-poisoning/src/train_mlsecops.py
Clean dataset: mlsecops-data-poisoning/data/iris_clean.csv
5% poisoned dataset: mlsecops-data-poisoning/data/iris_poisoned_5.csv
10% poisoned dataset: mlsecops-data-poisoning/data/iris_poisoned_10.csv
50% poisoned dataset: mlsecops-data-poisoning/data/iris_poisoned_50.csv
MLflow results: mlsecops-data-poisoning/results/metrics.csv
MLflow comparison screenshots: mlsecops-data-poisoning/screenshots/
