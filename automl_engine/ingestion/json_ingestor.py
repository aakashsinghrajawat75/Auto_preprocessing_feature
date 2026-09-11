import json
import pandas as pd
from typing import Union
from pathlib import Path
from automl_engine.ingestion.base import BaseDataIngestor


class JSONIngestor(BaseDataIngestor):
    """
    Concrete Data Ingestor for JSON & JSONL files.
    Supports nested JSON flattening into normalized tabular DataFrames.
    """

    def load_data(self, record_path: str = None, **kwargs) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"JSON file not found at path: {self.file_path}")

        try:
            # First attempt standard pandas read_json (works for records or jsonl)
            df = pd.read_json(self.file_path, **kwargs)
        except Exception:
            # Fallback to python json load + pd.json_normalize for complex nested JSON
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                # If root is a dictionary containing a list of records
                if record_path and record_path in data:
                    data = data[record_path]
                else:
                    # Find the first list value in dict if available
                    list_keys = [k for k, v in data.items() if isinstance(v, list)]
                    if list_keys:
                        data = data[list_keys[0]]
                    else:
                        data = [data]

            df = pd.json_normalize(data)

        self._update_metadata(df)
        return df
