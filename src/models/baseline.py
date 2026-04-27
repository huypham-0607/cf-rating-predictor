"""
Naive baselines and Ridge regression.

MeanPredictor and MedianPredictor follow the sklearn estimator API so they
work seamlessly in evaluation loops alongside tree models.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MeanPredictor(BaseEstimator, RegressorMixin):
    """Ignore X and return mean"""
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self
    
    def predict(self, X):
        return np.full(len(X), self.mean_)
    
class MedianPredictor(BaseEstimator, RegressorMixin):
    """Ignore X and return median"""
    def fit(self, X, y):
        self.median_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.median_)

def build_ridge(alpha: float = 10.0) -> Pipeline:
    """Ridge regression with StandardScaler — good linear baseline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])