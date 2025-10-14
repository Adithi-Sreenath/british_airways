import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RouteEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding="frequency", interaction=True, min_samples=5):
        
        self.encoding = encoding
        self.interaction = interaction
        self.min_samples = min_samples
        
    def fit(self, X, y=None):
        X = X.copy()
        X[['origin', 'destination']] = X['route'].str.split('-', expand=True)

        if self.encoding == "frequency":
            self.origin_map_ = X['origin'].value_counts().to_dict()
            self.dest_map_   = X['destination'].value_counts().to_dict()
            if self.interaction:
                X['origin_destination'] = X['origin'] + "_" + X['destination']
                self.inter_map_ = X['origin_destination'].value_counts().to_dict()

        elif self.encoding == "target":
            if y is None:
                raise ValueError("y must be provided for target encoding")
            df = X.assign(y=y)

            # global mean
            self.global_mean_ = df['y'].mean()

            # origin stats
            origin_stats = df.groupby('origin')['y'].agg(['mean','count'])
            origin_stats = origin_stats[origin_stats['count'] >= self.min_samples]
            self.origin_map_ = origin_stats['mean'].to_dict()

            # destination stats
            dest_stats = df.groupby('destination')['y'].agg(['mean','count'])
            dest_stats = dest_stats[dest_stats['count'] >= self.min_samples]
            self.dest_map_ = dest_stats['mean'].to_dict()

            # interaction stats
            if self.interaction:
                df['origin_destination'] = df['origin'] + "_" + df['destination']
                inter_stats = df.groupby('origin_destination')['y'].agg(['mean','count'])
                inter_stats = inter_stats[inter_stats['count'] >= self.min_samples]
                self.inter_map_ = inter_stats['mean'].to_dict()
        else:
            raise ValueError("encoding must be 'frequency' or 'target'")
        return self

    def transform(self, X):
        X = X.copy()
        X[['origin', 'destination']] = X['route'].str.split('-', expand=True)

        if self.encoding == "frequency":
            X['origin_enc'] = X['origin'].map(self.origin_map_).fillna(0)
            X['destination_enc'] = X['destination'].map(self.dest_map_).fillna(0)
            if self.interaction:
                X['origin_destination'] = X['origin'] + "_" + X['destination']
                X['origin_dest_enc'] = X['origin_destination'].map(self.inter_map_).fillna(0)

        elif self.encoding == "target":
            X['origin_enc'] = X['origin'].map(self.origin_map_).fillna(self.global_mean_)
            X['destination_enc'] = X['destination'].map(self.dest_map_).fillna(self.global_mean_)
            if self.interaction:
                X['origin_destination'] = X['origin'] + "_" + X['destination']
                X['origin_dest_enc'] = X['origin_destination'].map(self.inter_map_).fillna(self.global_mean_)
        
        return X.drop(columns=['origin','destination','origin_destination'], errors='ignore')
