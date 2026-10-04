# Model Card: IRIS Classifier

## 1. Model Overview

This model is a multiclass classifier trained on the IRIS dataset.

The model predicts one of three iris species:

- Setosa
- Versicolor
- Virginica

A Random Forest classifier is used for prediction.

## 2. Intended Use

The model is intended for educational and MLOps experimentation.

It demonstrates:

- Machine learning classification
- Fairness evaluation
- Explainability using SHAP
- Data drift monitoring
- Responsible ML governance

This model is not intended for safety-critical or real-world decision-making.

## 3. Training Data

The dataset contains 160 IRIS samples with four input features:

- sepal_length
- sepal_width
- petal_length
- petal_width

The target variable is `species`.

A randomly generated `location` attribute with values 0 and 1 was added only for fairness analysis.

The `location` attribute was not used as a model training feature.

## 4. Model

Model type:

Random Forest Classifier

Number of trees:

100

Random state:

42

## 5. Overall Performance

The model achieved the following overall test metrics:

| Metric | Score |
|---|---:|
| Accuracy | 0.9375 |
| Precision | 0.9479 |
| Recall | 0.9375 |

## 6. Fairness Evaluation

Fairness was evaluated using Fairlearn MetricFrame.

| Location | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| 0 | 0.9091 | 0.9273 | 0.9091 |
| 1 | 0.9524 | 0.9592 | 0.9524 |

The performance gaps were:

- Accuracy gap: 0.0433
- Precision gap: 0.0319
- Recall gap: 0.0433

The location attribute was randomly assigned, so the groups do not represent a real demographic population.

## 7. Explainability

SHAP was used to explain the model predictions.

SHAP summary plots were generated for:

- Setosa
- Versicolor
- Virginica

Positive SHAP values push the prediction toward the selected class, while negative SHAP values push it away.

Feature colors indicate feature values:

- Red indicates a high feature value.
- Blue indicates a low feature value.

The Virginica SHAP plot is used to understand which features contribute most strongly toward or away from Virginica predictions.

## 8. Drift Monitoring

A simulated production dataset was created by shifting:

- petal_length by +0.8
- petal_width by +0.3

The Kolmogorov-Smirnov test was used to compare training and production distributions.

| Feature | KS Statistic | Drift |
|---|---:|---|
| sepal_length | 0.0000 | No |
| sepal_width | 0.0000 | No |
| petal_length | 0.3688 | Yes |
| petal_width | 0.3250 | Yes |

A p-value below 0.05 was considered evidence of statistically significant distribution drift.

## 9. Limitations

- The dataset is small.
- The location attribute is randomly generated and does not represent a real protected demographic group.
- Fairness results may vary because the test set is small.
- Simulated drift does not represent every type of real-world production drift.
- Concept drift was not directly evaluated because ground-truth production labels were not available.

## 10. Fairness Considerations

The sensitive `location` attribute was excluded from model training.

It was retained only for auditing model performance across groups.

Fairness analysis should be repeated with real and meaningful sensitive attributes when deploying a model in a real-world setting.

## 11. Governance

The model should be monitored after deployment for:

- Data drift
- Model performance degradation
- Fairness gaps
- Changes in the data distribution

The model card documents the intended use, performance, fairness results, explainability approach, limitations, and monitoring considerations.

## 12. Summary

This project demonstrates responsible ML practices by combining:

- Model evaluation
- Fairness auditing with Fairlearn
- Explainability with SHAP
- Data drift monitoring
- Model documentation and governance
