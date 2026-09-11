import pandas as pd
from typing import Union
from pathlib import Path
from automl_engine.ingestion.base import BaseDataIngestor


class CSVIngestor(BaseDataIngestor):
    """
    Concrete Data Ingestor for CSV and TSV files.
    Handles encoding detection, custom delimiters, and raw messy string loading.
    """

    def load_data(self, delimiter: str = None, encoding: str = "utf-8", **kwargs) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found at path: {self.file_path}")

        # Attempt reading with specified delimiter or pandas autodetect
        try:
            if delimiter is not None:
                df = pd.read_csv(self.file_path, delimiter=delimiter, encoding=encoding, **kwargs)
            else:
                df = pd.read_csv(self.file_path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            # Fallback to latin-1 encoding if UTF-8 fails
            df = pd.read_csv(self.file_path, encoding="latin-1", **kwargs)

        self._update_metadata(df)
        return df
