import ulab
from ulab import numpy as np

class StandardScaler:
    """Standardizes features by removing mean and scaling to unit variance."""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float)
        
        # Calculate mean across rows (axis=0)
        self.mean = np.mean(X, axis=0)
        # Standard deviation: sqrt(mean((x - mean)^2))
        self.std = np.sqrt(np.mean((X - self.mean) ** 2, axis=0))
        
        # Prevent division by zero for constant features
        if len(self.std.shape) == 0:
            if self.std == 0:
                self.std = 1.0
        else:
            for i in range(len(self.std)):
                if self.std[i] == 0:
                    self.std[i] = 1.0
        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float)
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before transforming.")
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Transforms features by scaling each feature to a given range (default 0 to 1)."""
    def __init__(self, feature_range=(0, 1)):
        self.min_val = feature_range[0]
        self.max_val = feature_range[1]
        self.data_min = None
        self.data_max = None

    def fit(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float)
        
        self.data_min = np.min(X, axis=0)
        self.data_max = np.max(X, axis=0)
        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float)
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("MinMaxScaler must be fitted before transforming.")
        
        range_diff = self.data_max - self.data_min
        
        # Avoid division by zero
        if len(range_diff.shape) == 0:
            if range_diff == 0:
                range_diff = 1.0
        else:
            for i in range(len(range_diff)):
                if range_diff[i] == 0:
                    range_diff[i] = 1.0

        X_std = (X - self.data_min) / range_diff
        return X_std * (self.max_val - self.min_val) + self.min_val

    def fit_transform(self, X):
        return self.fit(X).transform(X)
