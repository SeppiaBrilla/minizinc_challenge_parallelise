from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class KmeansClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **hyperparams) -> None:
        self.model = KMeans(**hyperparams)

    def fit(self, X, Y):
        preds = self.model.fit_predict(X)
        cluster_vals = {}
        assert len(X) == len(Y)
        for i in range(len(preds)):
            if not preds[i] in cluster_vals:
                cluster_vals[preds[i]] = [0,0]
            cluster_vals[preds[i]][int(Y.iloc[i])] += 1
        self.labels = {cluster: np.argmax(vals) for cluster, vals in cluster_vals.items()}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.array(list(map(lambda x: self.labels[x], preds)))

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X))


