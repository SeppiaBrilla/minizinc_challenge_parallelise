from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Literal
import pandas as pd

class NoneScaler:
    def __init__(self) -> None:
        pass
    def fit(*args) -> None:
        pass

    def transform(self, data):
        return data

class PCAPreprocessor:
    def __init__(self, pca_type:str) -> None:
        if pca_type == 'None':
            self.pca = None
        else:
            n_components = pca_type if pca_type == 'mle' else int(pca_type)
            self.pca = PCA(n_components=n_components)

    def fit(self, X:pd.DataFrame):
        if not self.pca:
            return
        self.pca.fit(X.select_dtypes(include=['number']))

    def transform(self, X:pd.DataFrame):
        if not self.pca:
            return X
        non_numeric_X = X.select_dtypes(exclude=['number'])
        numeric_X = X.select_dtypes(include=['number'])
        components = self.pca.transform(numeric_X)
        comp_names = [f'comp_{i}' for i in range(len(components[0]))]
        pca_df = pd.DataFrame(components, columns=comp_names, index=X.index)
        return pd.concat([non_numeric_X, pca_df], axis=1)

def get_scaler(scaler_type:Literal['standard', 'minMax', 'None']) -> type:
    if scaler_type == 'standard':
        return StandardScaler
    if scaler_type == 'minMax':
        return MinMaxScaler
    return NoneScaler
