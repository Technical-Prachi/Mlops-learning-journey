# MLOps Learning Journey: From a Notebook to Production

An end-to-end project in which I took a simple Iris classifier and built the MLOps and LLMOps workflow around it, one layer at a time: data versioning, a feature store, CI/CD, experiment tracking, containers, Kubernetes, load testing, ML security, responsible AI, LLM fine-tuning and guardrails.

The model is intentionally simple. The point of this repository is the **engineering around the model**, not the model itself.

> Built during my BS in Data Science and Applications at IIT Madras (MLOps coursework), written up as a learning project.

## What This Demonstrates

- **Reproducibility:** data and model versioning (DVC), feature store (Feast), experiment tracking and model registry (MLflow)
- **Automation:** tests in CI, containerized API, Kubernetes deployment through GitHub Actions
- **Reliability:** load testing, autoscaling and bottleneck analysis
- **Responsible and secure ML:** data poisoning experiment, fairness, explainability, drift detection, a model card
- **LLMOps:** fine-tuning Gemini on Vertex AI, red teaming, input/output guardrails, audit logging

## Tech Stack

| Area | Tools |
|---|---|
| Data & model versioning | DVC, Google Cloud Storage |
| Feature management | Feast |
| Experiment tracking & registry | MLflow |
| Serving & deployment | FastAPI, Docker, Google Kubernetes Engine, Artifact Registry, HPA |
| CI/CD | GitHub Actions, pytest, CML |
| Load testing | wrk |
| Responsible AI | SHAP, Fairlearn, KS-test drift detection, model card |
| LLMOps | Vertex AI supervised fine-tuning (Gemini), custom guardrails |

## What Is Where

| Topic | Tools | Where to look |
|---|---|---|
| Training and inference pipeline | scikit-learn, Google Cloud Storage | [`train.py`](train.py), [`inference.py`](inference.py), [notes](learning-notes/training-and-inference-pipeline.md) |
| Data and model versioning | DVC | [notes](learning-notes/dvc-data-versioning.md) |
| Feature store | Feast | [`feast_feature_store/`](feast_feature_store/), [notes](learning-notes/feast-feature-store.md) |
| Continuous integration | pytest, GitHub Actions, CML | [`tests/`](tests/), [`ci-cd-reference/`](ci-cd-reference/), [notes](learning-notes/ci-pytest-github-actions.md) |
| Experiment tracking | MLflow | [notes](learning-notes/mlflow-experiment-tracking.md) |
| API, container, Kubernetes deployment | FastAPI, Docker, GKE | [`app.py`](app.py), [`Dockerfile`](Dockerfile), [`deployment.yaml`](deployment.yaml), [notes](learning-notes/docker-fastapi-gke-deployment.md) |
| Load testing and autoscaling | wrk, HPA | [`load-testing-wrk-results/`](load-testing-wrk-results/), [`hpa.yaml`](hpa.yaml), [notes](learning-notes/load-testing-and-autoscaling.md) |
| ML security | Data poisoning, MLflow | [`mlsecops-data-poisoning/`](mlsecops-data-poisoning/) |
| Responsible AI | SHAP, Fairlearn, drift tests, model card | [`explainability-fairness-drift/`](explainability-fairness-drift/) |
| LLM fine-tuning | Vertex AI, Gemini | [`llmops-gemini-finetuning/`](llmops-gemini-finetuning/) |
| LLM guardrails | Red teaming, custom guardrails | [`llm-guardrails-red-teaming/`](llm-guardrails-red-teaming/) |

The order above is the order in which I built things.

## Results at a Glance

### Baseline model
Random Forest on Iris, 93.3% accuracy on the held-out split (`artifacts/metrics.txt`).

### Load testing (wrk, 30 s, 4 threads)

| Connections | Requests/sec | Avg latency | Timeouts |
|---|---|---|---|
| 1000 | 80.1 | 1.13 s | 413 |
| 2000 | 106.8 | 1.64 s | 3090 |

The service became capacity-constrained under heavy load. I compared autoscaling up to 3 pods against a cap of 1 pod; the capped setup saturated CPU and produced more timeouts. Raw outputs: [`load-testing-wrk-results/`](load-testing-wrk-results/).

### Data poisoning

| Poisoned training data | Accuracy |
|---|---|
| 0% | 96.88% |
| 5% | 96.88% |
| 10% | 96.88% |
| 50% | 65.63% |

The model tolerated low levels of random corruption on this small dataset, then dropped by about 31 points at 50%. Caveat: random poisoning of a tiny dataset is not a model of real targeted attacks.

### Fairness, explainability and drift
- Overall accuracy 93.75%; accuracy gap of 0.043 between two randomly assigned location groups. Because the groups are random, this demonstrates the method rather than a real fairness finding.
- SHAP plots for all three classes. KS-test drift detection flagged `petal_length` and `petal_width` after a simulated shift.
- A model card: [`explainability-fairness-drift/MODEL_CARD.md`](explainability-fairness-drift/MODEL_CARD.md).

### Fine-tuning Gemini on Iris

| Metric | V1 raw features | V2 text description |
|---|---|---|
| Accuracy | 56.67% | 36.67% |
| Strict format compliance | 0% | 0% |

Both fine-tuned models did far worse than the simple Random Forest, and often returned explanatory text instead of a single class name. Takeaway: a general LLM is the wrong tool for small tabular classification, and LLM evaluation must include output-format compliance, not only accuracy.

### LLM guardrails

| Metric | Result |
|---|---|
| Prompt-injection attempts blocked | 12 / 12 |
| Prompt-leakage attempts blocked | 12 / 12 |
| False positive rate | 0% |
| V1 accuracy: baseline to guarded | 56.67% to 96.67% |
| V2 accuracy: baseline to guarded | 36.67% to 80.00% |

Caveat: the red-team set is small (12 prompts per attack type), and the accuracy gain comes largely from normalizing the model's output. It shows the approach, not production-grade security.

## Repository Structure

```
.
├── train.py, inference.py            # training and inference (Feast + MLflow)
├── app.py, Dockerfile                # FastAPI service and container
├── deployment.yaml, hpa.yaml         # Kubernetes manifests (project ID is a placeholder)
├── feast_feature_store/              # Feast feature store
├── tests/                            # pytest data and model tests
├── data/iris.csv                     # dataset used by the tests
├── artifacts/                        # saved model and metrics
├── ci-cd-reference/                  # GitHub Actions workflows (reference only)
├── learning-notes/                   # short notes on the first stages of the project
├── load-testing-wrk-results/         # wrk load-test outputs
├── mlsecops-data-poisoning/          # data poisoning experiment
├── explainability-fairness-drift/    # SHAP, Fairlearn, drift, model card
├── llmops-gemini-finetuning/         # Gemini fine-tuning evaluation
└── llm-guardrails-red-teaming/       # guardrails and red teaming
```

## Running It

The cloud parts (GCS, GKE, Vertex AI, the DVC remote) ran in my own Google Cloud project, which is now closed, so those scripts will not run as they are. The data-only and local parts can be run like this:

```bash
pip install -r requirements.txt
pytest -v                                       # run from the repository root
python mlsecops-data-poisoning/src/poison_data.py   # regenerate the poisoned datasets
```

`artifacts/model.pkl` was saved with scikit-learn 1.3.2. If loading it fails on a newer version, install `scikit-learn==1.3.2` or retrain the model.

The GitHub Actions workflows are kept in `ci-cd-reference/` for reference. They are not active, because they relied on cloud credentials that no longer exist.

## Security Note

An early version of this work committed a service account key to a public repository. Google's scanner detected it and disabled the key. I rebuilt the project in this repository with a clean history and no credentials. For CI, the right approach is Workload Identity Federation instead of key files, which is what I used in my later project.

## Related Projects

- **Heart Disease MLOps on GKE:** SHAP, Fairlearn, PSI drift monitoring, CI/CD with Workload Identity Federation
- **Enterprise PDF RAG Assistant**, **RAG Experimentation Lab**, **AI Resume Analyzer:** see my GitHub profile

## Author

**Prachi Kushwaha**, BS Data Science and Applications, IIT Madras
[GitHub](https://github.com/Technical-Prachi) | [LinkedIn](https://www.linkedin.com/in/YOUR-LINKEDIN-ID)
