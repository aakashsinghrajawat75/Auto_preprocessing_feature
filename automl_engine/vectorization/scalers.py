import pandas as pd
import numpy as np
from typing import List, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class NumericalScaler:
    """
    Scaler for Numerical Features.
    Supports: 'standard', 'minmax', 'robust'.
    """

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.scaler = None
        self.num_cols: List[str] = []

    def fit(self, df: pd.DataFrame, num_cols: List[str] = None) -> "NumericalScaler":
        if num_cols is None:
            self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.num_cols = num_cols

        if not self.num_cols:
            return self

        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        self.scaler.fit(df[self.num_cols].fillna(0.0))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.num_cols or self.scaler is None:
            return np.empty((len(df), 0))

        return self.scaler.transform(df[self.num_cols].fillna(0.0))
