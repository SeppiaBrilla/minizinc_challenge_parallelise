from typing import Literal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

class Closer(BaseEstimator, ClassifierMixin):
    def __init__(self, n_features:int, _type:Literal['mean','median'], _sorting_metric:Literal['same', 'variance']) -> None:
        self.n_features = n_features
        self._type = _type
        self._sorting_metric = _sorting_metric

    def fit(self, X:pd.DataFrame, y:pd.DataFrame):
        X_negative = X[y == 0]
        X_positive = X[y == 1]
        cols = X.columns
        cols_vals_positive = {}
        cols_vals_negative = {}
        for col in cols:
            _score_p = X_positive[col].mean() if self._type == "mean" else X_positive[col].median()
            _score_n = X_negative[col].mean() if self._type == "mean" else X_positive[col].median()
            cols_vals_positive[col] = (_score_p, X_positive[col].var() if self._sorting_metric == "variance" else np.mean([np.abs(_score_p - X_positive.iloc[i][col]) for i in range(len(X_positive))]))
            cols_vals_negative[col] = (_score_n, X_positive[col].var() if self._sorting_metric == "variance" else np.mean([np.abs(_score_n - X_positive.iloc[i][col]) for i in range(len(X_negative))]))
        self.positive = {k:v[0] for k,v in sorted(cols_vals_positive.items(), key=lambda x: x[1][1])[:self.n_features]}
        self.negative = {k:v[0] for k,v in sorted(cols_vals_negative.items(), key=lambda x: x[1][1])[:self.n_features]}

    def predict(self, X:pd.DataFrame):
        positive_results = []
        negative_results = []
        for i in range(len(X)):
            el = X.iloc[i]
            pos_arr = []
            neg_arr = []
            for col, v in self.positive.items():
                pos_arr.append((el[col] - v) ** 2)
            for col, v in self.negative.items():
                neg_arr.append((el[col] - v) ** 2)
            positive_results.append(sum(pos_arr) / len(pos_arr))
            negative_results.append(sum(neg_arr) / len(neg_arr))
        return pd.DataFrame([1 if positive_results[i] < negative_results[i] else 0 for i in range(len(positive_results))], columns=['y'])

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X))
