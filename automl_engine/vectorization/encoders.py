import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class CategoricalEncoder:
    """
    Encoder for Categorical Features.
    Supports: 'onehot', 'ordinal'.
    """

    def __init__(self, encoding_type: str = "onehot", handle_unknown: str = "ignore"):
        self.encoding_type = encoding_type
        self.handle_unknown = handle_unknown
        self.encoder = None
        self.cat_cols: List[str] = []

    def fit(self, df: pd.DataFrame, cat_cols: List[str] = None) -> "CategoricalEncoder":
        if cat_cols is None:
            self.cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            self.cat_cols = cat_cols

        if not self.cat_cols:
            return self

        cat_df = df[self.cat_cols].astype(str)

        if self.encoding_type == "onehot":
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown=self.handle_unknown)
        elif self.encoding_type == "ordinal":
            self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

        self.encoder.fit(cat_df)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.cat_cols or self.encoder is None:
            return np.empty((len(df), 0))

        cat_df = df[self.cat_cols].astype(str)
        return self.encoder.transform(cat_df)
