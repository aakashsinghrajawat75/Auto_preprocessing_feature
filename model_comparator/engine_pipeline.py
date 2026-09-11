import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple
from automl_engine.pipeline import AutoMLPipeline


class EnginePipeline:
    """
    SIH AutoML Engine Feature Engineering CSV Pipeline.
    - Zero-touch Regex Unit Extraction ("1,450 sq.ft." -> 1450.0, "85 Lakhs" -> 85.0)
    - KNN Smart Missing Value Imputation
    - Multi-Column Vectorization Engine (128-d dense tensor)
    - Exports automl_processed.csv
    """

    def __init__(self, target_dim: int = 128, imputation_strategy: str = "knn"):
        self.automl_pipeline = AutoMLPipeline(
            target_dim=target_dim,
            imputation_strategy=imputation_strategy
        )

    def fit_transform(
        self,
        file_path_or_df: Union[str, Path, pd.DataFrame],
        record_path: str = None
    ) -> Tuple[pd.DataFrame, np.ndarray, Path]:
        """
        Executes zero-touch AutoML feature processing and exports automl_processed.csv.
        """
        if isinstance(file_path_or_df, pd.DataFrame):
            source_stem = "custom_data"
        else:
            source_stem = Path(file_path_or_df).stem

        cleaned_df, v_dense, report = self.automl_pipeline.fit_transform(
            file_path_or_df, record_path=record_path
        )

        output_dir = Path(__file__).resolve().parent.parent / "output" / source_stem
        output_dir.mkdir(parents=True, exist_ok=True)
        export_file = output_dir / "automl_processed.csv"
        cleaned_df.to_csv(export_file, index=False)

        return cleaned_df, v_dense, export_file
