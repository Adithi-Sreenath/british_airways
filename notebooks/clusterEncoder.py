import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

class OriginClusterEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, origin_col="booking_origin", lead_col="purchase_lead_winsor", 
                 bins=[0, 3, 7, 14, 30, 60, 90, 180, 365], n_clusters=5, random_state=42):
        self.origin_col = origin_col
        self.lead_col = lead_col
        self.bins = bins
        self.n_clusters = n_clusters
        self.random_state = random_state

        # fitted attributes
        self.kmeans_ = None
        self.origin_to_cluster_ = {}
        self.bin_intervals_ = None

    def fit(self, X, y=None):
        # 1. Define bin intervals consistently
        purchase_lead_bin = pd.cut(X[self.lead_col], bins=self.bins, include_lowest=True)
        self.bin_intervals_ = purchase_lead_bin.cat.categories  # store categories

        # 2. Distribution per origin (row-normalized histograms)
        dist = pd.crosstab(X[self.origin_col], purchase_lead_bin, normalize="index").fillna(0)

        # 3. Fit KMeans
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto")
        clusters = self.kmeans_.fit_predict(dist)

        # 4. Store mapping origin → cluster
        self.origin_to_cluster_ = dict(zip(dist.index, clusters))

        return self

    def transform(self, X):
        X = X.copy()

        # map known origins
        X[self.origin_col + "_cluster"] = X[self.origin_col].map(self.origin_to_cluster_)

        # handle unseen origins
        unseen_mask = X[self.origin_col + "_cluster"].isna()
        if unseen_mask.any():
            centroids = self.kmeans_.cluster_centers_

            for origin in X.loc[unseen_mask, self.origin_col].unique():
                subset = X.loc[X[self.origin_col] == origin, self.lead_col]

                # build normalized histogram over same bins as training
                lead_bin = pd.cut(subset, bins=self.bins, include_lowest=True)
                hist = (
                    lead_bin.value_counts(normalize=True, sort=False)
                    .reindex(self.bin_intervals_, fill_value=0)
                    .to_numpy()
                )

                # assign to nearest centroid
                cluster = np.argmin(np.linalg.norm(centroids - hist, axis=1))
                self.origin_to_cluster_[origin] = cluster

            # fill missing
            X[self.origin_col + "_cluster"] = X[self.origin_col].map(self.origin_to_cluster_)

        return X[[self.origin_col + "_cluster"]]
