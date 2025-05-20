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
