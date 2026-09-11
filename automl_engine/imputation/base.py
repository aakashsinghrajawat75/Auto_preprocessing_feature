from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class BaseImputer(ABC):
    """
    Abstract Base Class for Missing Value Imputers.
    Follows scikit-learn API design (fit_transform, fit, transform).
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseImputer":
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
