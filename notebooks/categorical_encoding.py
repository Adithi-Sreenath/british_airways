import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.compose import ColumnTransformer 

# --- Custom Transformer for Frequency Encoding ---
class FrequencyEncoder(BaseEstimator, TransformerMixin):     
    def __init__(self, col):         
        self.col = col         
        self.freq_map = None      
    
    def fit(self, X, y=None):         
        self.freq_map = X[self.col].value_counts().to_dict()        
        return self     
    
    def transform(self, X):         
        X = X.copy()         
        X[self.col + "_freq"] = X[self.col].map(self.freq_map).fillna(0)         
        return X[[self.col + "_freq"]]  

# --- Custom Transformer for Cyclical Encoding ---         
class CyclicalEncoder(BaseEstimator, TransformerMixin):     
    def __init__(self, col, max_val):         
        self.col = col         
        self.max_val = max_val      
    
    def fit(self, X, y=None):         
        return self      
    
    def transform(self, X):         
        X = X.copy()         
        X[self.col + "_sin"] = np.sin(2 * np.pi * X[self.col] / self.max_val)         
        X[self.col + "_cos"] = np.cos(2 * np.pi * X[self.col] / self.max_val)         
        return X[[self.col + "_sin", self.col + "_cos"]]    

# --- Factory function to build preprocessor --- 
def make_preprocessor(cat_low_card, cat_high_card, cyclical_col, cyclical_max):
    transformers = [
        ("low_card_cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_low_card),
    ]
    
    # Add each high cardinality column separately
    for i, col in enumerate(cat_high_card):
        transformers.append((f"high_card_cat_{i}", FrequencyEncoder(col), [col]))
    
    # Add cyclical encoder
    transformers.append(("cyclical", CyclicalEncoder(cyclical_col, max_val=cyclical_max), [cyclical_col]))
    
    preprocessor = ColumnTransformer(         
        transformers=transformers,         
        remainder="drop"    
    )      
    return preprocessor