import pandas as pd
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class TFIDFTextVectorizer:
    """
    Vectorizer for Unstructured Text Columns.
    Uses TF-IDF feature extraction combined with TruncatedSVD for dense embedding synthesis.
    """

    def __init__(self, max_features: int = 500, n_components: int = 32):
        self.max_features = max_features
        self.n_components = n_components
        self.vectorizers = {}
        self.svd_models = {}
        self.text_cols: List[str] = []

    def fit(self, df: pd.DataFrame, text_cols: List[str]) -> "TFIDFTextVectorizer":
        self.text_cols = text_cols
        if not self.text_cols:
            return self

        for col in self.text_cols:
            texts = df[col].fillna("").astype(str).tolist()
            vec = TfidfVectorizer(max_features=self.max_features, stop_words="english")
            tfidf_mat = vec.fit_transform(texts)

            actual_components = min(self.n_components, tfidf_mat.shape[1] - 1) if tfidf_mat.shape[1] > 1 else 1
            svd = TruncatedSVD(n_components=actual_components, random_state=42)
            svd.fit(tfidf_mat)

            self.vectorizers[col] = vec
            self.svd_models[col] = svd

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.text_cols or not self.vectorizers:
            return np.empty((len(df), 0))

        embeddings = []
        for col in self.text_cols:
            texts = df[col].fillna("").astype(str).tolist()
            vec = self.vectorizers[col]
            svd = self.svd_models[col]

            tfidf_mat = vec.transform(texts)
            emb = svd.transform(tfidf_mat)
            embeddings.append(emb)

        return np.hstack(embeddings)
