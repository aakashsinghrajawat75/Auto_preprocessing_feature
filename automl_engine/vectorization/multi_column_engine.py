import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from automl_engine.vectorization.scalers import NumericalScaler
from automl_engine.vectorization.encoders import CategoricalEncoder
from automl_engine.vectorization.text_vectorizer import TFIDFTextVectorizer


class MultiColumnVectorizationEngine:
    """
    Multi-Column Relationship Mining & Dense Vector Synthesis Engine.
    Combines numerical, categorical, and textual features into a compact,
    normalized, dense numerical vector embedding V_dense (e.g. 128-dimensional matrix),
    injectable directly into PyTorch, XGBoost, or Scikit-Learn.
    """

    def __init__(self, target_dim: int = 128, scaler_type: str = "standard", encoding_type: str = "onehot"):
        self.target_dim = target_dim
        self.scaler = NumericalScaler(scaler_type=scaler_type)
        self.encoder = CategoricalEncoder(encoding_type=encoding_type)
        self.text_vectorizer = TFIDFTextVectorizer(n_components=32)
        self.pca = None
        self.final_scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, column_types: Dict[str, List[str]]) -> "MultiColumnVectorizationEngine":
        num_cols = column_types.get("numerical", [])
        cat_cols = column_types.get("categorical", [])
        text_cols = column_types.get("text", [])

        # Fit sub-vectorizers
        if num_cols:
            self.scaler.fit(df, num_cols)
        if cat_cols:
            self.encoder.fit(df, cat_cols)
        if text_cols:
            self.text_vectorizer.fit(df, text_cols)

        # Transform to generate concatenated feature matrix for PCA fitting
        concatenated_matrix = self._build_concatenated_matrix(df, num_cols, cat_cols, text_cols)

        num_samples, total_features = concatenated_matrix.shape

        if total_features > 0:
            actual_dim = min(self.target_dim, num_samples, total_features)
            if actual_dim > 0 and total_features > actual_dim:
                self.pca = PCA(n_components=actual_dim, random_state=42)
                reduced_matrix = self.pca.fit_transform(concatenated_matrix)
            else:
                self.pca = None
                reduced_matrix = concatenated_matrix

            self.final_scaler.fit(reduced_matrix)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, column_types: Dict[str, List[str]]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("MultiColumnVectorizationEngine must be fitted before calling transform.")

        num_cols = column_types.get("numerical", [])
        cat_cols = column_types.get("categorical", [])
        text_cols = column_types.get("text", [])

        concatenated_matrix = self._build_concatenated_matrix(df, num_cols, cat_cols, text_cols)

        if concatenated_matrix.shape[1] == 0:
            return np.zeros((len(df), self.target_dim), dtype=np.float32)

        if self.pca is not None:
            dense_matrix = self.pca.transform(concatenated_matrix)
        else:
            dense_matrix = concatenated_matrix

        # Normalize final embedding matrix
        normalized_dense_matrix = self.final_scaler.transform(dense_matrix)

        # Pad to target_dim if features < target_dim
        if normalized_dense_matrix.shape[1] < self.target_dim:
            pad_width = self.target_dim - normalized_dense_matrix.shape[1]
            padding = np.zeros((len(df), pad_width), dtype=normalized_dense_matrix.dtype)
            normalized_dense_matrix = np.hstack([normalized_dense_matrix, padding])

        return normalized_dense_matrix

    def _build_concatenated_matrix(self, df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], text_cols: List[str]) -> np.ndarray:
        matrices = []

        if num_cols:
            num_mat = self.scaler.transform(df)
            if num_mat.shape[1] > 0:
                matrices.append(num_mat)

        if cat_cols:
            cat_mat = self.encoder.transform(df)
            if cat_mat.shape[1] > 0:
                matrices.append(cat_mat)

        if text_cols:
            text_mat = self.text_vectorizer.transform(df)
            if text_mat.shape[1] > 0:
                matrices.append(text_mat)

        if not matrices:
            return np.empty((len(df), 0))

        return np.hstack(matrices)
