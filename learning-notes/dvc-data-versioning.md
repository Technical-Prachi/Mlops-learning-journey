# Data and Model Versioning with DVC

**Goal:** Version datasets and models without storing large files in Git.

**What I built**
- Initialized DVC and configured a Google Cloud Storage bucket as the remote.
- Tracked the Iris dataset (`data/iris.csv.dvc`) and the trained model with DVC; Git tracked only the small pointer files.
- Created two versions of the data and model, and restored earlier versions using `git checkout` and `dvc checkout`.

**What I learned**
- Git is for code; data and models need their own versioning.
- A Git commit plus a DVC pointer gives a reproducible snapshot of code, data and model together.

**Note:** The DVC remote bucket was in a cloud project that is now closed, so `dvc pull` no longer works.
