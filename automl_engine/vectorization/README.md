# 🧬 `automl_engine/vectorization` — Multi-Column Dense Vector Synthesis

The `vectorization` module converts preprocessed heterogeneous tabular features into a compact, normalized 128-dimensional dense vector matrix ($V_{\text{dense}}$).

---

## 📄 File Inventory & Class Responsibilities

### 1. `scalers.py`
- **`NumericalScaler`**: Standardizes numerical features using `StandardScaler` ($\mu=0, \sigma=1$), `MinMaxScaler`, or `RobustScaler`.

### 2. `encoders.py`
- **`CategoricalEncoder`**: Encodes categorical features using One-Hot Encoding or Ordinal Encoding.

### 3. `text_vectorizer.py`
- **`TFIDFTextVectorizer`**: Applies TF-IDF vectorization and TruncatedSVD dimensionality reduction for text fields ($D=32$).

### 4. `multi_column_engine.py`
- **`MultiColumnVectorizationEngine`**:
  - Concatenates scaled numerical, one-hot categorical, and TF-IDF text matrices.
  - Applies PCA dimensionality reduction or padding to generate a normalized dense matrix:
    $$V_{\text{dense}} \in \mathbb{R}^{N \times 128}$$

---

## 💻 Code Example

```python
from automl_engine.vectorization import MultiColumnVectorizationEngine

engine = MultiColumnVectorizationEngine(target_dim=128)
engine.fit(df_imputed, column_types)
v_dense = engine.transform(df_imputed, column_types)

# v_dense is a numpy ndarray of shape (N, 128)
```
