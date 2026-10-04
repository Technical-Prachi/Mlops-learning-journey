# Feature Store with Feast

**Goal:** Use one source of truth for features in both training and inference.

**What I built** (see `feast_feature_store/`)
- Prepared the Iris data for Feast (ID column, target, event timestamp, Parquet file).
- Defined an Entity, a File Source and a Feature View.
- Ran `feast apply` and materialized features into the SQLite online store.
- `train.py` reads historical features from the offline store; `inference.py` reads online features.

**What I learned**
- The difference between the offline store (training) and the online store (serving).
- Reading features from one definition prevents training-serving skew.
