import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformMixin

class ZRPEngineeringBase(BaseEstimator, TransformMixin):
    """Base class for feature engineering"""
    def __init__(self, *args, **kwargs):
        self.zrp = True
    def fit(self, X, y):
        return self
    def transform(self, X):
        pass