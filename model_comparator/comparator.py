import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Optional

from model_comparator.baseline_pipeline import BaselinePipeline
from model_comparator.engine_pipeline import EnginePipeline
from automl_engine.evaluation.model_evaluator import ModelEvaluator


class ModelComparator:
    """
    Dedicated Model Comparator orchestrator.
    Step 1: Runs BaselinePipeline -> generates baseline_processed.csv
    Step 2: Runs EnginePipeline -> generates automl_processed.csv & V_dense
    Step 3: Evaluates Linear Model, Decision Tree, Random Forest, KNN, XGBoost
            Computing R2, MSE, MAE, RMSE for Regression and Accuracy, Precision, Recall, F1 for Classification.
    """

    def __init__(self, target_dim: int = 128, imputation_strategy: str = "knn", random_state: int = 42):
        self.target_dim = target_dim
        self.imputation_strategy = imputation_strategy
        self.evaluator = ModelEvaluator(random_state=random_state)

    def run_comparison(
        self,
        file_path_or_df: Union[str, Path, pd.DataFrame],
        target_column: str,
        record_path: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs step-by-step dual CSV generation and ML evaluation comparison.
        """
        # Step 1: Baseline Naive EDA -> baseline_processed.csv
        baseline_pipe = BaselinePipeline(target_column=target_column)
        baseline_df, baseline_csv_path = baseline_pipe.fit_transform(file_path_or_df, record_path=record_path)

        # Step 2: SIH AutoML Engine -> automl_processed.csv
        engine_pipe = EnginePipeline(target_dim=self.target_dim, imputation_strategy=self.imputation_strategy)
        cleaned_df, v_dense, engine_csv_path = engine_pipe.fit_transform(file_path_or_df, record_path=record_path)

        # Step 3: Comparative Evaluation
        if isinstance(file_path_or_df, pd.DataFrame):
            raw_df = file_path_or_df.copy()
        else:
            from automl_engine.ingestion.factory import DataIngestorFactory
            ingestor = DataIngestorFactory.get_ingestor(file_path_or_df)
            raw_df = ingestor.load_data(record_path=record_path) if record_path else ingestor.load_data()

        eval_report = self.evaluator.compare_pipelines(
            raw_df=raw_df,
            cleaned_df=cleaned_df,
            target_col=target_column,
            v_dense=v_dense,
            task_type=task_type
        )

        return {
            "target_column": target_column,
            "task_type": eval_report["task_type"],
            "metric_type": eval_report["metric_type"],
            "baseline_csv_path": str(baseline_csv_path),
            "automl_csv_path": str(engine_csv_path),
            "baseline_df": baseline_df,
            "automl_df": cleaned_df,
            "v_dense": v_dense,
            "baseline_metrics": eval_report["baseline_metrics"],
            "automl_metrics": eval_report["automl_metrics"],
            "comparison_table": eval_report["comparison_table"],
            "best_baseline_model": eval_report["best_baseline_model"],
            "best_automl_model": eval_report["best_automl_model"],
            "overall_avg_improvement_pct": eval_report["overall_avg_improvement_pct"]
        }
