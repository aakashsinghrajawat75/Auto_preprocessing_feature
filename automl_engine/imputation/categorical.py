import numpy as np
import pandas as pd
from typing import List
from sklearn.impute import SimpleImputer
from automl_engine.imputation.base import BaseImputer


class CategoricalImputer(BaseImputer):
    """
    Imputer for Categorical and Text Features.
    Supports strategies: 'most_frequent', 'constant'.
    """

    def __init__(self, strategy: str = "most_frequent", fill_value: str = "Missing"):
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer = None
        self.cat_cols: List[str] = []

    def _prepare_cat_df(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_df = df[self.cat_cols].copy()
        for col in self.cat_cols:
            cat_df[col] = cat_df[col].apply(
                lambda x: np.nan if pd.isna(x) or str(x).strip().lower() in ["nan", "none", "<na>", "null", ""] else str(x)
            )
        return cat_df

    def fit(self, df: pd.DataFrame, cat_cols: List[str] = None) -> "CategoricalImputer":
        if cat_cols is None:
            self.cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            self.cat_cols = cat_cols

        if not self.cat_cols:
            return self

        cat_df = self._prepare_cat_df(df)

        if self.strategy == "most_frequent":
            self.imputer = SimpleImputer(strategy="most_frequent", missing_values=np.nan)
        elif self.strategy == "constant":
            self.imputer = SimpleImputer(strategy="constant", fill_value=self.fill_value, missing_values=np.nan)
        else:
            raise ValueError(f"Unknown categorical imputation strategy: {self.strategy}")

        self.imputer.fit(cat_df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if not self.cat_cols or self.imputer is None:
            return df_out

        cat_df = self._prepare_cat_df(df_out)
        imputed_array = self.imputer.transform(cat_df)
        df_out[self.cat_cols] = pd.DataFrame(imputed_array, columns=self.cat_cols, index=df_out.index)
        return df_out
