# Continuous Integration with pytest and GitHub Actions

**Goal:** Automatically test data and model quality on every push.

**What I built**
- `tests/test_data.py`: checks that the dataset exists, has no missing values, has the expected columns, and that feature values are in valid ranges.
- `tests/test_model.py`: checks that the saved model reaches at least 90% accuracy.
- A GitHub Actions workflow (`ci-cd-reference/ci.yml`) that installs dependencies, runs pytest on every push and pull request, and posts the test report on pull requests using CML.
- In the original run, 5 tests passed. The `dvc pull` step is commented out in the final workflow file.

**What I learned**
- ML projects need tests for data and models, not only for code.
- CI feedback on pull requests catches problems before merging.
