# Explainability, Fairness, and Drift in the IRIS Pipeline

## Overview

This assignment demonstrates responsible machine learning practices using the IRIS classification dataset.

The main objectives are:

- Adding a sensitive attribute for fairness analysis
- Evaluating fairness using Fairlearn
- Explaining model predictions using SHAP
- Detecting data drift in simulated production data
- Documenting the model using a model card

## Project Structure

explainability-fairness-drift/
├── README.md
├── MODEL_CARD.md
├── data/
│   ├── iris_with_location.csv
│   └── iris_production.csv
├── results/
│   ├── fairness_metrics.csv
│   ├── drift_results.csv
│   ├── shap_setosa.png
│   ├── shap_versicolor.png
│   ├── shap_virginica.png
│   ├── drift_sepal_length.png
│   ├── drift_sepal_width.png
│   ├── drift_petal_length.png
│   └── drift_petal_width.png
└── src/
    ├── create_sensitive_attribute.py
    ├── fairness.py
    ├── explainability.py
    └── drift.py

## Task 1 - Sensitive Attribute

A location column was randomly generated with values 0 and 1.

The location attribute was used only as a sensitive attribute for fairness analysis.

It was not used as a model training feature.

The dataset contains 160 samples.

Location distribution:

- Location 0: 75 samples
- Location 1: 85 samples

## Task 2 - Fairness Analysis

Fairness was evaluated using Fairlearn MetricFrame.

The following metrics were calculated:

- Accuracy
- Precision
- Recall

### Overall Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.9375 |
| Precision | 0.9479 |
| Recall | 0.9375 |

### Performance by Location

| Location | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| 0 | 0.9091 | 0.9273 | 0.9091 |
| 1 | 0.9524 | 0.9592 | 0.9524 |

Performance gaps:

- Accuracy gap: 0.0433
- Precision gap: 0.0319
- Recall gap: 0.0433

Since location was randomly assigned, it does not represent a real demographic group.

## Task 3 - SHAP Explainability

SHAP was used to explain the Random Forest classifier.

Summary plots were generated for all three classes:

- Setosa
- Versicolor
- Virginica

The generated plots are available in the results directory.

### Virginica Interpretation

A positive SHAP value means that a feature pushes the model toward predicting Virginica.

A negative SHAP value means that a feature pushes the prediction away from Virginica.

In the SHAP summary plot:

- The right side represents positive SHAP values.
- The left side represents negative SHAP values.
- Red dots represent high feature values.
- Blue dots represent low feature values.

## Task 4 - Data Drift

A simulated production dataset was created by shifting:

- petal_length by +0.8
- petal_width by +0.3

The Kolmogorov-Smirnov test was used to compare the training and production distributions.

### Drift Results

| Feature | KS Statistic | Drift |
|---|---:|---|
| sepal_length | 0.0000 | No |
| sepal_width | 0.0000 | No |
| petal_length | 0.3688 | Yes |
| petal_width | 0.3250 | Yes |

A p-value below 0.05 was considered evidence of statistically significant drift.

The results show that petal_length and petal_width experienced significant distribution changes.

## Data Drift vs Concept Drift

Data drift means that the distribution of input features changes over time.

Concept drift means that the relationship between input features and the target variable changes over time.

Data drift can be monitored using statistical tests on feature distributions.

Concept drift requires monitoring model predictions against actual ground-truth labels.

## Task 5 - Model Card

A model card is included in MODEL_CARD.md.

It documents:

- Intended use
- Training data
- Model details
- Performance metrics
- Fairness results
- Explainability approach
- Drift monitoring
- Limitations
- Governance considerations

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Fairlearn
- SHAP
- SciPy
- Matplotlib
- Git and GitHub
- Google Cloud Vertex AI Workbench

## Responsible ML Summary

This assignment demonstrates that a machine learning model should not only be accurate.

A production ML system should also consider:

1. Explainability
2. Fairness
3. Data drift
4. Model performance
5. Governance
6. Model limitations

These practices help make ML systems more trustworthy and easier to monitor in production.
