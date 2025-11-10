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

def get_scaler(scaler_type:Literal['standard', 'minMax', 'None']) -> type:
    if scaler_type == 'standard':
        return StandardScaler
    if scaler_type == 'minMax':
        return MinMaxScaler
    return NoneScaler
