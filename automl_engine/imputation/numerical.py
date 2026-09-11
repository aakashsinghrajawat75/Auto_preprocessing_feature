import pandas as pd
import numpy as np
from typing import List, Union
from sklearn.impute import SimpleImputer, KNNImputer
from automl_engine.imputation.base import BaseImputer


class StatisticalImputer(BaseImputer):
    """
    Imputer for Numerical Features.
    Supports strategies: 'median', 'mean', 'knn', 'iterative'.
    """

    def __init__(self, strategy: str = "knn", n_neighbors: int = 5):
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.imputer = None
        self.num_cols: List[str] = []

    def fit(self, df: pd.DataFrame, num_cols: List[str] = None) -> "StatisticalImputer":
        if num_cols is None:
            self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.num_cols = num_cols

        if not self.num_cols:
            return self

        num_df = df[self.num_cols]

        if self.strategy in ["mean", "median"]:
            self.imputer = SimpleImputer(strategy=self.strategy)
            self.imputer.fit(num_df)
        elif self.strategy == "knn":
            self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
            self.imputer.fit(num_df)
        elif self.strategy == "iterative":
            try:
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                self.imputer = IterativeImputer(random_state=42)
                self.imputer.fit(num_df)
            except Exception:
                # Fallback to KNN if IterativeImputer fails
                self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
                self.imputer.fit(num_df)
        else:
            raise ValueError(f"Unknown numerical imputation strategy: {self.strategy}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if not self.num_cols or self.imputer is None:
            return df_out

        imputed_array = self.imputer.transform(df_out[self.num_cols])
        df_out[self.num_cols] = pd.DataFrame(imputed_array, columns=self.num_cols, index=df_out.index)
        return df_out
