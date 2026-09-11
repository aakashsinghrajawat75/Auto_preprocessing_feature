from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Union
from pathlib import Path


class BaseDataIngestor(ABC):
    """
    Abstract Base Class for Data Ingestors adhering to OOP principles.
    Defines the contract for loading CSV, Excel, JSON, XML datasets into standardized DataFrames.
    """

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self._metadata: Dict[str, Any] = {
            "file_name": self.file_path.name,
            "file_extension": self.file_path.suffix.lower(),
            "raw_size_bytes": self.file_path.stat().st_size if self.file_path.exists() else 0,
            "loaded_rows": 0,
            "loaded_columns": 0
        }

    @abstractmethod
    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Loads data from file_path and returns a standardized pandas DataFrame.
        Must be implemented by concrete subclasses.
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns metadata about the ingested file and loaded dataset.
        """
        return self._metadata.copy()

    def _update_metadata(self, df: pd.DataFrame) -> None:
        """
        Updates metadata after successful DataFrame ingestion.
        """
        self._metadata["loaded_rows"] = len(df)
        self._metadata["loaded_columns"] = len(df.columns)
        self._metadata["column_names"] = list(df.columns)
