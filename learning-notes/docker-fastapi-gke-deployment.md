# FastAPI, Docker and Kubernetes (GKE) Deployment

**Goal:** Serve the model as an API and deploy it automatically.

**What I built**
- `app.py`: a FastAPI service with `GET /` (health message) and `POST /predict` (takes four Iris measurements, returns the predicted class).
- `Dockerfile` to containerize the API.
- `deployment.yaml` for Google Kubernetes Engine (resource requests and limits set). The service was exposed through a LoadBalancer; that manifest is not included in this repository.
- A GitHub Actions workflow (`ci-cd-reference/cd.yml`) that builds the image, pushes it to Artifact Registry and deploys to GKE.

**What I learned**
- The same container image runs the same way everywhere.
- Deployment should be automated and driven by Git.

**Note:** The cloud project is closed, so the workflow is kept as reference only. The project ID in the files is replaced with a placeholder.
