## Taxi Recommender (GCP Vertex AI)

### Goal
Recommend taxi company based on pickup/dropoff coordinates using logistic regression.

### Steps
1. `python data/download_sample.py`
2. `python src/model/train.py`
3. Upload `model.pkl` to GCS
4. `python src/deploy/deploy_model.py`