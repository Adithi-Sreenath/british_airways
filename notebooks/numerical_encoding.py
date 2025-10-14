from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import numpy as np
class CyclicalHourEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, col="flight_hour", max_val=24):
        self.col = col
        self.max_val = max_val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.col + "_sin"] = np.sin(2 * np.pi * X[self.col] / self.max_val)
        X[self.col + "_cos"] = np.cos(2 * np.pi * X[self.col] / self.max_val)
        return X[[self.col + "_sin", self.col + "_cos"]]
def make_preprocessor_numerical(cyclical_col="flight_hour", cyclical_max=24):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cyclical", CyclicalHourEncoder(cyclical_col, cyclical_max), [cyclical_col]),
        ],
        remainder="drop"
    )

    return preprocessor