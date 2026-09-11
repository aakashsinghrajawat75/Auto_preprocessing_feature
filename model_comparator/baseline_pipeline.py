import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from automl_engine.ingestion.factory import DataIngestorFactory


class BaselinePipeline:
    """
    Simulates Naive / Manual EDA & Feature Engineering.
    - Standard ingestion
    - Simple median fill for numeric columns
    - Naive string label encoding without regex numeric unit extraction
    - Exports baseline_processed.csv
    """

    def __init__(self, target_column: Optional[str] = None):
        self.target_column = target_column
        self.label_encoders = {}

    def fit_transform(
        self,
        file_path_or_df: Union[str, Path, pd.DataFrame],
        record_path: str = None
    ) -> Tuple[pd.DataFrame, Path]:
        """
        Executes baseline naive feature processing and exports baseline_processed.csv.
        """
        if isinstance(file_path_or_df, pd.DataFrame):
            df_raw = file_path_or_df.copy()
            source_stem = "custom_data"
        else:
            path_obj = Path(file_path_or_df)
            source_stem = path_obj.stem
            ingestor = DataIngestorFactory.get_ingestor(path_obj)
            df_raw = ingestor.load_data(record_path=record_path) if record_path else ingestor.load_data()

        df_processed = df_raw.copy()

        # Separate numeric vs categorical without regex parsing
        num_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df_processed.columns if c not in num_cols]

        # Numeric Simple Imputation (Median)
        if num_cols:
            num_imp = SimpleImputer(strategy="median")
            df_processed[num_cols] = num_imp.fit_transform(df_processed[num_cols])

        # Naive Categorical Encoding (Label Encoding without Regex unit parsing)
        for col in cat_cols:
            if self.target_column and col == self.target_column:
                continue
            series_str = df_processed[col].astype(str).fillna("Missing")
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(series_str)
            self.label_encoders[col] = le

        # Ensure output folder exists
        output_dir = Path(__file__).resolve().parent.parent / "output" / source_stem
        output_dir.mkdir(parents=True, exist_ok=True)
        export_file = output_dir / "baseline_processed.csv"
        df_processed.to_csv(export_file, index=False)

        return df_processed, export_file
