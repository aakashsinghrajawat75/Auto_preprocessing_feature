import pandas as pd
from typing import Union
from pathlib import Path
from automl_engine.ingestion.base import BaseDataIngestor


class ExcelIngestor(BaseDataIngestor):
    """
    Concrete Data Ingestor for Excel files (.xlsx, .xls).
    """

    def load_data(self, sheet_name: Union[str, int] = 0, **kwargs) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Excel file not found at path: {self.file_path}")

        df = pd.read_excel(self.file_path, sheet_name=sheet_name, **kwargs)
        self._update_metadata(df)
        return df
