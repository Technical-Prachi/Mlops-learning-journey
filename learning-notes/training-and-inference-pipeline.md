# Training and Inference Pipeline

**Goal:** Turn a notebook-style Iris classifier into a repeatable pipeline.

**What I built**
- `train.py`: loads data, trains a Random Forest, evaluates it, and uploads the model and metrics to Google Cloud Storage with a timestamped path.
- `inference.py`: downloads the latest model from cloud storage and predicts on a sample input.
- `requirements.txt` for reproducible setup.

**What I learned**
- Keep training and inference as separate scripts.
- Store model artifacts outside the code repository, with versioned paths.

**Note:** The final versions of these scripts (with Feast and MLflow added in later weeks) are in the repository root.
