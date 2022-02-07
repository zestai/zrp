class SetKey():
    """
    This class sets a key as the index
    
    Parameters
    ----------
    key: str 
        Key to set as index. If not provided, a key will be generated.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y):
        return self
        
    def transform(self, X):
        key_in_cols = self.key in X.columns
        if key_in_cols:
            X = X.set_index(self.key)
        return X

