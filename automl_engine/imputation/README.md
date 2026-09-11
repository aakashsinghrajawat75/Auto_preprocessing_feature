# 🩹 `automl_engine/imputation` — Missing Value Imputation Layer

The `imputation` module handles missing values across numerical and categorical features.

---

## 📄 File Inventory & Class Responsibilities

### 1. `base.py`
- **`BaseImputer`**: Abstract interface specifying `fit()`, `transform()`, and `fit_transform()`.

### 2. `numerical.py`
- **`StatisticalImputer`**:
  - Implements numerical imputation strategies:
    - `'knn'`: K-Nearest Neighbors Imputation ($K=5$) based on Euclidean distance.
    - `'median'`: Fills NaNs with feature median.
    - `'mean'`: Fills NaNs with feature mean.
    - `'iterative'`: MICE multivariate imputation.

### 3. `categorical.py`
- **`CategoricalImputer`**:
  - Implements categorical imputation strategies:
    - `'most_frequent'`: Fills NaNs with feature mode.
    - `'constant'`: Fills NaNs with `'Missing'` sentinel token.

---

## 💻 Code Example

```python
from automl_engine.imputation import StatisticalImputer, CategoricalImputer

# Impute numeric columns
num_imp = StatisticalImputer(strategy="knn")
df_imputed = num_imp.fit_transform(df)

# Impute categorical columns
cat_imp = CategoricalImputer(strategy="most_frequent")
df_imputed = cat_imp.fit_transform(df_imputed)
```
