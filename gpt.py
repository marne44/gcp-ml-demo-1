# Directory Structure (scaffold)

# taxi-ml/
# ├── notebooks/
# │   └── eda.ipynb               # For EDA and sampling
# ├── data/
# │   └── download_sample.py      # Sample and preprocess raw data
# ├── src/
# │   ├── data_prep/
# │   │   └── features.py         # Feature engineering
# │   ├── model/
# │   │   ├── train.py            # Model training
# │   │   └── evaluate.py         # Metrics
# │   ├── deploy/
# │   │   └── deploy_model.py    # Vertex AI deployment
# │   └── monitor/
# │       └── log_predictions.py     # Basic logging
# ├── pipelines/                # (Optional for later)
# ├── Dockerfile                # For custom training job (optional)
# ├── requirements.txt          # Pip dependencies
# └── README.md

# Example: data/download_sample.py
import pandas as pd


def download_sample(output_path: str, n_rows: int = 10000):
    url = "https://data.cityofchicago.org/api/views/wrvz-psew/rows.csv?accessType=DOWNLOAD"
    df = pd.read_csv(url, nrows=n_rows)
    df.to_csv(output_path, index=False)


# Example: src/data_prep/features.py
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class HaversineTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from math import radians, cos, sin, asin, sqrt

        coords = X[
            [
                "Pickup Centroid Latitude",
                "Pickup Centroid Longitude",
                "Dropoff Centroid Latitude",
                "Dropoff Centroid Longitude",
            ]
        ].values
        dist = []
        for row in coords:
            lat1, lon1, lat2, lon2 = map(radians, row)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            r = 6371  # km
            dist.append(c * r)
        return np.array(dist).reshape(-1, 1)


# Example: src/model/train.py
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.data_prep.features import HaversineTransformer

if __name__ == "__main__":
    df = pd.read_csv("data/sample.csv")
    X = df[
        [
            "Pickup Centroid Latitude",
            "Pickup Centroid Longitude",
            "Dropoff Centroid Latitude",
            "Dropoff Centroid Longitude",
        ]
    ]
    y = df["Taxi Company"]

    pipe = Pipeline([("dist", HaversineTransformer()), ("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(X, y)
    joblib.dump(pipe, "model.pkl")


# Example: src/deploy/deploy_model.py
from google.cloud import aiplatform


def deploy_model():
    aiplatform.init(project="YOUR_PROJECT", location="us-central1")
    model = aiplatform.Model.upload(
        display_name="taxi-classifier",
        artifact_uri="gs://YOUR_BUCKET/models/taxi/",
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",
    )
    endpoint = model.deploy(machine_type="n1-standard-2")
    print("Model deployed to:", endpoint.resource_name)


# requirements.txt
pandas
scikit - learn
google - cloud - aiplatform
joblib

# README.md (summary)
"""
## Taxi Recommender (GCP Vertex AI)

### Goal
Recommend taxi company based on pickup/dropoff coordinates using logistic regression.

### Steps
1. `python data/download_sample.py`
2. `python src/model/train.py`
3. Upload `model.pkl` to GCS
4. `python src/deploy/deploy_model.py`
"""
