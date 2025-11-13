from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd

class KnnClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k:int) -> None:
        self.k = k

    def fit(self, X, Y):
        self.X = X.to_numpy(dtype=float)
        self.Y = Y

    def predict(self, X:pd.DataFrame, t=False) -> np.ndarray:
        preds = []
        for x in X.to_numpy(dtype=float):
            x_dists = []
            for i, _x in enumerate(self.X):
                x_dists.append((np.sum((x - _x ) ** 2), i))
            x_dists = sorted(x_dists, key = lambda x: x[0])
            if t:
                print(x_dists[:self.k])
            y_vals = [self.Y.iloc[idx] for _, idx in x_dists[:self.k]]
            counts = (y_vals.count(0), y_vals.count(1))
            preds.append(np.argmax(counts))
        return np.array(preds)


