import pandas as pd
import numpy as np
from typing import Dict, Any, List


class AutoEDAEngine:
    """
    Automated Exploratory Data Analysis (EDA) Engine.
    Provides autonomous dataset profiling, type detection, missingness analysis,
    statistical summaries, and correlation matrices.
    """

    def __init__(self, text_cardinality_threshold: int = 20):
        self.text_cardinality_threshold = text_cardinality_threshold

    def profile_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes zero-touch profiling on the input DataFrame.
        Returns a comprehensive EDA report dictionary.
        """
        total_rows = len(df)
        total_cols = len(df.columns)

        column_types = self._detect_column_types(df)
        missing_summary = self._analyze_missingness(df)
        stats_summary = self._compute_statistics(df, column_types)
        correlation_matrix = self._compute_correlations(df, column_types["numerical"])

        report = {
            "dataset_overview": {
                "total_rows": total_rows,
                "total_columns": total_cols,
                "total_missing_cells": int(df.isna().sum().sum()),
                "overall_missing_percentage": float(round((df.isna().sum().sum() / (total_rows * total_cols)) * 100, 2)) if total_rows * total_cols > 0 else 0.0
            },
            "column_classification": column_types,
            "missingness_summary": missing_summary,
            "statistical_summary": stats_summary,
            "correlation_matrix": correlation_matrix
        }

        return report

    def _detect_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        numerical = []
        categorical = []
        text = []
        datetime_cols = []

        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                numerical.append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                datetime_cols.append(col)
            else:
                # String / Object column - safely cast unhashable elements (dicts/lists) to string
                str_series = df[col].apply(lambda x: str(x) if isinstance(x, (dict, list)) else x)
                num_unique = str_series.nunique(dropna=True)
                if num_unique > self.text_cardinality_threshold and str_series.astype(str).str.len().mean() > 15:
                    text.append(col)
                else:
                    categorical.append(col)

        return {
            "numerical": numerical,
            "categorical": categorical,
            "text": text,
            "datetime": datetime_cols
        }

    def _analyze_missingness(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        total_rows = len(df)
        missing_info = {}

        for col in df.columns:
            null_count = int(df[col].isna().sum())
            null_ratio = float(round((null_count / total_rows) * 100, 2)) if total_rows > 0 else 0.0
            missing_info[col] = {
                "null_count": null_count,
                "null_percentage": null_ratio,
                "has_missing": null_count > 0
            }

        return missing_info

    def _compute_statistics(self, df: pd.DataFrame, column_types: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        stats = {}

        # Numerical columns statistics
        for col in column_types["numerical"]:
            s = df[col].dropna()
            if len(s) > 0:
                stats[col] = {
                    "type": "numerical",
                    "mean": float(round(s.mean(), 4)),
                    "std": float(round(s.std(), 4)) if len(s) > 1 else 0.0,
                    "min": float(s.min()),
                    "25%": float(s.quantile(0.25)),
                    "median": float(s.median()),
                    "75%": float(s.quantile(0.75)),
                    "max": float(s.max()),
                    "skewness": float(round(s.skew(), 4)) if len(s) > 2 else 0.0
                }

        # Categorical columns statistics
        for col in column_types["categorical"]:
            s = df[col].dropna().apply(lambda x: str(x) if isinstance(x, (dict, list)) else x)
            if len(s) > 0:
                stats[col] = {
                    "type": "categorical",
                    "cardinality": int(s.nunique()),
                    "top_mode": str(s.mode()[0]) if not s.mode().empty else None,
                    "mode_frequency": int(s.value_counts().iloc[0]) if not s.empty else 0
                }

        # Text columns statistics
        for col in column_types["text"]:
            s = df[col].dropna().astype(str)
            word_counts = s.str.split().str.len()
            stats[col] = {
                "type": "text",
                "sample_count": len(s),
                "avg_word_count": float(round(word_counts.mean(), 2)) if len(s) > 0 else 0.0,
                "max_word_count": int(word_counts.max()) if len(s) > 0 else 0
            }

        return stats

    def _compute_correlations(self, df: pd.DataFrame, num_cols: List[str]) -> Dict[str, Dict[str, float]]:
        if len(num_cols) < 2:
            return {}

        corr_df = df[num_cols].corr().fillna(0.0)
        corr_dict = {}
        for col in corr_df.columns:
            corr_dict[col] = {k: float(round(v, 4)) for k, v in corr_df[col].items()}

        return corr_dict
