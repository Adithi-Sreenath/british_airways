import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 cat_high_card=None,
                 route_col=None,
                 cyclical_cols=None,   # dict: {"flight_hour": 24, "flight_day": 7}
                 numeric_cols=None,
                 target_encode_cols=None):
        
        self.cat_high_card = cat_high_card
        self.route_col = route_col
        self.cyclical_cols = cyclical_cols if cyclical_cols is not None else {}
        self.numeric_cols = numeric_cols if numeric_cols is not None else []
        self.target_encode_cols = target_encode_cols if target_encode_cols is not None else []

        # initialize storage
        self.freq_map = {}
        self.te_map = {}

    def fit(self, X, y=None):
        X = X.copy()

        # --- Frequency Encoding ---
        if self.cat_high_card:
            self.freq_map[self.cat_high_card] = X[self.cat_high_card].value_counts(normalize=True)
        if self.route_col:
            self.freq_map[self.route_col] = X[self.route_col].value_counts(normalize=True)

        # --- Target Encoding ---
        if y is not None:
            X_with_target = X.copy()
            X_with_target["target"] = y.values if hasattr(y, 'values') else y
            for col in self.target_encode_cols:
                te_map = X_with_target.groupby(col)["target"].mean()
                self.te_map[col] = te_map

        return self

    def transform(self, X):
        X = X.copy()

        # --- Frequency Encoding ---
        if self.cat_high_card and self.cat_high_card in self.freq_map:
            X[self.cat_high_card + "_freq"] = X[self.cat_high_card].map(self.freq_map[self.cat_high_card]).fillna(0)
        if self.route_col and self.route_col in self.freq_map:
            X[self.route_col + "_freq"] = X[self.route_col].map(self.freq_map[self.route_col]).fillna(0)

        # --- Target Encoding ---
        for col, mapping in self.te_map.items():
            X[col + "_te"] = X[col].map(mapping).fillna(mapping.mean())

        # --- Cyclical Encoding ---
        for col, max_val in self.cyclical_cols.items():
            X[col + "_sin"] = np.sin(2 * np.pi * X[col] / max_val)
            X[col + "_cos"] = np.cos(2 * np.pi * X[col] / max_val)

        # --- Collect final features ---
        final_cols = []
        if self.cat_high_card and (self.cat_high_card + "_freq") in X.columns:
            final_cols.append(self.cat_high_card + "_freq")
        if self.route_col and (self.route_col + "_freq") in X.columns:
            final_cols.append(self.route_col + "_freq")
        
        final_cols += [col + "_te" for col in self.te_map.keys()]
        for col in self.cyclical_cols.keys():
            final_cols += [col + "_sin", col + "_cos"]
        final_cols += self.numeric_cols

        return X[final_cols]



    
