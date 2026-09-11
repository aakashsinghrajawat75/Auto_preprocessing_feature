import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Union

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class ModelEvaluator:
    """
    Model Evaluator for Comparative Benchmarking between:
    1. Baseline Manual EDA (Standard naive preprocessing, simple imputation, label encoding)
    2. SIH AutoML Engine (Regex sanitization, smart KNN imputation, dense vector synthesis)
    
    Evaluates across 5 ML Model Families:
      - Linear Models (Linear / Logistic Regression)
      - Decision Trees
      - Random Forests
      - K-Nearest Neighbors (KNN)
      - XGBoost (XGBRegressor / XGBClassifier with Gradient Boosting fallback)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    @staticmethod
    def _extract_numeric_target(series: pd.Series) -> pd.Series:
        """Helper to extract numbers from messy target strings like '85 Lakhs' or '$1,200'."""
        cleaned = series.astype(str).str.replace(",", "", regex=False)
        extracted = cleaned.str.extract(r"([-+]?\d*\.?\d+)")[0]
        numeric_series = pd.to_numeric(extracted, errors="coerce")
        return numeric_series

    def auto_detect_task_type(self, y: pd.Series) -> str:
        """
        Auto-detects whether the target variable represents a Regression or Classification task.
        """
        non_null_y = y.dropna()
        if non_null_y.empty:
            return "classification"

        # Try numeric conversion
        numeric_y = pd.to_numeric(non_null_y, errors="coerce")
        if numeric_y.isna().mean() > 0.5:
            # High non-numeric ratio -> classification
            return "classification"

        unique_count = numeric_y.nunique()
        if unique_count > 10 or (numeric_y.dtype in [float, np.float64, np.float32] and unique_count > 5):
            return "regression"
        else:
            return "classification"

    def _prepare_target(self, y_raw: pd.Series, task_type: str) -> Tuple[np.ndarray, Optional[LabelEncoder]]:
        """Prepares target vector and returns target label encoder if classification."""
        if task_type == "regression":
            y_num = self._extract_numeric_target(y_raw)
            if y_num.isna().all():
                y_num = pd.to_numeric(y_raw, errors="coerce")
            valid_mask = ~y_num.isna()
            return y_num.values, valid_mask, None
        else:
            y_str = y_raw.astype(str).fillna("Missing")
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_str)
            valid_mask = np.ones(len(y_encoded), dtype=bool)
            return y_encoded, valid_mask, le

    def _build_baseline_features(self, raw_df: pd.DataFrame, target_col: str) -> np.ndarray:
        """
        Simulates standard manual EDA & simple feature encoding without AutoML regex sanitization.
        - Simple median imputer for numeric columns
        - Simple Label Encoding for string columns
        - StandardScaler normalization
        """
        feature_df = raw_df.drop(columns=[target_col], errors="ignore").copy()
        processed_cols = []

        for col in feature_df.columns:
            series = feature_df[col]
            if pd.api.types.is_numeric_dtype(series):
                imp = SimpleImputer(strategy="median")
                col_vals = imp.fit_transform(series.values.reshape(-1, 1))
                processed_cols.append(col_vals)
            else:
                # Naive string label encoding without regex parsing
                str_series = series.astype(str).fillna("Missing")
                le = LabelEncoder()
                encoded_vals = le.fit_transform(str_series).reshape(-1, 1)
                processed_cols.append(encoded_vals)

        if not processed_cols:
            return np.zeros((len(raw_df), 1))

        X_baseline = np.hstack(processed_cols)
        scaler = StandardScaler()
        return scaler.fit_transform(X_baseline)

    def _get_regression_models(self) -> Dict[str, Any]:
        models = {
            "Linear Model": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=self.random_state),
            "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=6, random_state=self.random_state),
            "KNN": KNeighborsRegressor(n_neighbors=5),
        }
        if HAS_XGBOOST:
            models["XGBoost"] = XGBRegressor(n_estimators=50, max_depth=4, random_state=self.random_state, verbosity=0)
        else:
            models["XGBoost"] = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=self.random_state)
        return models

    def _get_classification_models(self) -> Dict[str, Any]:
        models = {
            "Linear Model": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.random_state),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=6, class_weight="balanced", random_state=self.random_state),
            "KNN": KNeighborsClassifier(n_neighbors=5),
        }
        if HAS_XGBOOST:
            models["XGBoost"] = XGBClassifier(n_estimators=50, max_depth=4, random_state=self.random_state, eval_metric="logloss", verbosity=0)
        else:
            models["XGBoost"] = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=self.random_state)
        return models

    def _evaluate_models_on_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str
    ) -> Dict[str, Dict[str, float]]:
        """Trains models on features X and target y, returning evaluation metrics dictionary."""
        results = {}

        # Use train-test split (80/20) or fallback to 100% train if dataset < 5 rows
        if len(X) >= 6:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        models = self._get_regression_models() if task_type == "regression" else self._get_classification_models()

        for name, model in models.items():
            try:
                # Adjust KNN n_neighbors if sample size is tiny
                if name == "KNN":
                    n_samples = len(X_train)
                    model.n_neighbors = min(5, max(1, n_samples - 1))

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if task_type == "regression":
                    r2 = float(r2_score(y_test, y_pred))
                    mse = float(mean_squared_error(y_test, y_pred))
                    rmse = float(np.sqrt(mse))
                    mae = float(mean_absolute_error(y_test, y_pred))

                    # Standardized primary score: R2
                    results[name] = {
                        "primary_score": round(max(r2, -1.0), 4),
                        "R2_Score": round(r2, 4),
                        "RMSE": round(rmse, 4),
                        "MAE": round(mae, 4),
                        "MSE": round(mse, 4)
                    }
                else:
                    acc = float(accuracy_score(y_test, y_pred))
                    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
                    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
                    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

                    results[name] = {
                        "primary_score": round(acc, 4),
                        "Accuracy": round(acc, 4),
                        "Precision": round(prec, 4),
                        "Recall": round(rec, 4),
                        "F1_Score": round(f1, 4)
                    }
            except Exception as e:
                # Return placeholder metric on exception
                results[name] = {
                    "primary_score": 0.0,
                    "error": str(e)
                }

        return results

    def compare_pipelines(
        self,
        raw_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        target_col: str,
        v_dense: Optional[np.ndarray] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end benchmarking comparison between Baseline Manual EDA and AutoML Engine features.

        :param raw_df: Original un-sanitized raw DataFrame.
        :param cleaned_df: Cleaned DataFrame from AutoMLPipeline.
        :param target_col: Target column name.
        :param v_dense: Synthesized dense vector matrix V_dense (optional, will use numeric features if None).
        :param task_type: 'regression' or 'classification' (auto-detected if None).
        :return: Comprehensive comparison dictionary.
        """
        if target_col not in raw_df.columns and target_col not in cleaned_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        y_series = cleaned_df[target_col] if target_col in cleaned_df.columns else raw_df[target_col]

        if task_type is None:
            task_type = self.auto_detect_task_type(y_series)

        # Prepare target vector
        y, valid_mask, _ = self._prepare_target(y_series, task_type)

        # 1. Evaluate Baseline Manual EDA Features
        X_baseline = self._build_baseline_features(raw_df, target_col)
        X_baseline_valid = X_baseline[valid_mask]
        y_valid = y[valid_mask]

        baseline_metrics = self._evaluate_models_on_dataset(X_baseline_valid, y_valid, task_type)

        # 2. Evaluate AutoML Engine Features
        if v_dense is not None and len(v_dense) == len(raw_df):
            X_automl = v_dense
        else:
            # Use numeric features from cleaned_df
            feat_df = cleaned_df.drop(columns=[target_col], errors="ignore").select_dtypes(include=[np.number])
            if feat_df.empty:
                X_automl = X_baseline
            else:
                X_automl = StandardScaler().fit_transform(feat_df.values)

        X_automl_valid = X_automl[valid_mask]
        automl_metrics = self._evaluate_models_on_dataset(X_automl_valid, y_valid, task_type)

        # 3. Compute Comparison Summary & Improvement Deltas
        summary_table = []
        metric_key = "R2_Score" if task_type == "regression" else "Accuracy"

        for model_name in baseline_metrics.keys():
            b_score = baseline_metrics[model_name].get("primary_score", 0.0)
            a_score = automl_metrics.get(model_name, {}).get("primary_score", 0.0)

            # Compute percentage delta
            if b_score == a_score:
                delta_pct = 0.0
            elif b_score <= 0 and a_score > b_score:
                delta_pct = round((a_score - b_score) * 100.0, 2)
            elif b_score > 0:
                delta_pct = round(((a_score - b_score) / abs(b_score)) * 100.0, 2)
            else:
                delta_pct = 0.0

            status = "🚀 Improved" if delta_pct > 0 else ("⚡ Equal" if delta_pct == 0 else "🔻 Lower")

            summary_table.append({
                "Model": model_name,
                f"Baseline ({metric_key})": b_score,
                f"AutoML Engine ({metric_key})": a_score,
                "Delta Improvement (%)": delta_pct,
                "Status": status
            })

        # Best Model Selections
        best_baseline = max(baseline_metrics.items(), key=lambda x: x[1].get("primary_score", -999))
        best_automl = max(automl_metrics.items(), key=lambda x: x[1].get("primary_score", -999))

        overall_gain = round(
            sum(row["Delta Improvement (%)"] for row in summary_table) / len(summary_table), 2
        ) if summary_table else 0.0

        return {
            "task_type": task_type,
            "target_column": target_col,
            "metric_type": metric_key,
            "baseline_metrics": baseline_metrics,
            "automl_metrics": automl_metrics,
            "comparison_table": summary_table,
            "best_baseline_model": {
                "model": best_baseline[0],
                "score": best_baseline[1].get("primary_score", 0.0)
            },
            "best_automl_model": {
                "model": best_automl[0],
                "score": best_automl[1].get("primary_score", 0.0)
            },
            "overall_avg_improvement_pct": overall_gain
        }
