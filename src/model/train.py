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
