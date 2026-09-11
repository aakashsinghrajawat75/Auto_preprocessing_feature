import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional
from automl_engine.ingestion.factory import DataIngestorFactory
from automl_engine.sanitization.regex_sanitizer import RegexSanitizer
from automl_engine.eda.profiler import AutoEDAEngine
from automl_engine.imputation.numerical import StatisticalImputer
from automl_engine.imputation.categorical import CategoricalImputer
from automl_engine.vectorization.multi_column_engine import MultiColumnVectorizationEngine
from automl_engine.evaluation.model_evaluator import ModelEvaluator
from automl_engine.utils.exporters import TensorExporter


class AutoMLPipeline:
    """
    Unified Zero-Touch AutoML Preprocessing, EDA & Vectorization Pipeline.
    Orchestrates:
      1. Ingestion (CSV, Excel, JSON, XML) via factory pattern
      2. Auto-Sanitization (Regex-based numeric extraction from messy strings)
      3. Autonomous EDA Profiling (Metadata, statistics, missingness, correlation)
      4. Smart Missing Value Imputation (KNN / Statistical + Categorical Mode/Constant)
      5. Multi-Column Dense Vector Synthesis (Generates 128-d numerical vector matrix V_dense)
      6. Tensor & File Export (NumPy .npy, Parquet, PyTorch Tensor, CSV)
      7. Model Evaluation & Benchmarking (Baseline vs AutoML Engine on Linear, Tree, RF, KNN, XGBoost)
    """

    def __init__(
        self,
        target_dim: int = 128,
        imputation_strategy: str = "knn",
        scaler_type: str = "standard"
    ):
        self.target_dim = target_dim
        self.sanitizer = RegexSanitizer()
        self.eda_engine = AutoEDAEngine()
        self.num_imputer = StatisticalImputer(strategy=imputation_strategy)
        self.cat_imputer = CategoricalImputer(strategy="most_frequent")
        self.vectorizer = MultiColumnVectorizationEngine(target_dim=target_dim, scaler_type=scaler_type)
        self.evaluator = ModelEvaluator()

        self.last_eda_report: Dict[str, Any] = {}
        self.last_sanitization_report: Dict[str, Any] = {}
        self.last_ingestion_metadata: Dict[str, Any] = {}
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.v_dense: Optional[np.ndarray] = None

    def fit_transform(
        self,
        file_path_or_df: Union[str, Path, pd.DataFrame],
        record_path: str = None
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
        """
        Executes zero-touch end-to-end preprocessing, EDA, imputation, and vectorization.

        :param file_path_or_df: Path to CSV, Excel, JSON, XML file OR a raw Pandas DataFrame.
        :param record_path: Optional JSON/XML sub-path tag.
        :return: Tuple of (cleaned_df, v_dense_matrix, full_pipeline_report)
        """
        # Step 1: Ingestion
        if isinstance(file_path_or_df, pd.DataFrame):
            raw_df = file_path_or_df.copy()
            self.last_ingestion_metadata = {
                "source": "dataframe",
                "loaded_rows": len(raw_df),
                "loaded_columns": len(raw_df.columns)
            }
        else:
            ingestor = DataIngestorFactory.get_ingestor(file_path_or_df)
            raw_df = ingestor.load_data(record_path=record_path) if record_path else ingestor.load_data()
            self.last_ingestion_metadata = ingestor.get_metadata()

        self.raw_df = raw_df.copy()

        # Step 2: Sanitization (Regex numeric extraction)
        sanitized_df, sanitization_report = self.sanitizer.sanitize_dataframe(raw_df)
        self.last_sanitization_report = sanitization_report

        # Step 3: Automated EDA Profiling
        eda_report = self.eda_engine.profile_dataframe(sanitized_df)
        self.last_eda_report = eda_report

        col_types = eda_report["column_classification"]

        # Step 4: Missing Value Imputation
        df_imputed = sanitized_df.copy()
        if col_types["numerical"]:
            df_imputed = self.num_imputer.fit_transform(df_imputed)
        if col_types["categorical"]:
            df_imputed = self.cat_imputer.fit_transform(df_imputed)

        self.cleaned_df = df_imputed.copy()

        # Step 5: Multi-Column Dense Vector Synthesis (V_dense)
        self.vectorizer.fit(df_imputed, col_types)
        v_dense = self.vectorizer.transform(df_imputed, col_types)
        self.v_dense = v_dense

        # Build Pipeline Execution Summary Report
        pipeline_report = {
            "ingestion_metadata": self.last_ingestion_metadata,
            "sanitization_report": self.last_sanitization_report,
            "eda_report": eda_report,
            "vectorization_summary": {
                "input_rows": len(df_imputed),
                "v_dense_shape": list(v_dense.shape),
                "target_vector_dim": self.target_dim
            }
        }

        return df_imputed, v_dense, pipeline_report

    def evaluate_models(
        self,
        target_column: str,
        raw_df: Optional[pd.DataFrame] = None,
        cleaned_df: Optional[pd.DataFrame] = None,
        v_dense: Optional[np.ndarray] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs comparative benchmarking between Baseline Manual EDA and AutoML Engine features.
        """
        r_df = raw_df if raw_df is not None else self.raw_df
        c_df = cleaned_df if cleaned_df is not None else self.cleaned_df
        v_mat = v_dense if v_dense is not None else self.v_dense

        if r_df is None or c_df is None:
            raise ValueError("Pipeline must be fit using fit_transform before evaluating models, or raw_df/cleaned_df must be provided.")

        return self.evaluator.compare_pipelines(
            raw_df=r_df,
            cleaned_df=c_df,
            target_col=target_column,
            v_dense=v_mat,
            task_type=task_type
        )

    def export_results(
        self,
        df_cleaned: pd.DataFrame,
        v_dense: np.ndarray,
        output_dir: Union[str, Path]
    ) -> Dict[str, str]:
        """
        Exports clean tabular data, dense vector matrix, and fitted pipeline model to output directory.
        """
        return TensorExporter.export_all(
            df_cleaned, v_dense, self.last_eda_report, output_dir, pipeline_model=self
        )

    def save_model(self, filepath: Union[str, Path]) -> str:
        """
        Saves the fitted AutoMLPipeline instance to a .pkl pickle file.
        """
        return TensorExporter.save_pickle(self, filepath)

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> "AutoMLPipeline":
        """
        Loads a saved AutoMLPipeline instance from a .pkl pickle file.
        """
        return TensorExporter.load_pickle(filepath)

