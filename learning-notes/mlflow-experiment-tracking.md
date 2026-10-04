# Experiment Tracking with MLflow

**Goal:** Make experiments comparable and reproducible.

**What I built**
- Hyperparameter tuning over `n_estimators` (50, 100) and `max_depth` (3, 5): four runs in total.
- Each run logs parameters, accuracy and the model artifact to MLflow.
- Registered the model as `iris-classifier` in the MLflow Model Registry.
- `inference.py` loads the model from MLflow. DVC tracking of the model was removed in favor of the registry.

**What I learned**
- Logged parameters and metrics make results comparable and reproducible.
- A model registry gives models names and versions, which DVC pointers alone do not.
