# 📊 `automl_engine/eda` — Autonomous EDA Profiler Module

The `eda` module performs autonomous Exploratory Data Analysis (EDA) on sanitised DataFrames.

---

## 📄 File Inventory & Class Responsibilities

### `profiler.py`
- **`AutoEDAEngine`**:
  - Classifies columns into: `numerical`, `categorical`, `text`, `datetime`.
  - Calculates summary statistics (mean, std, min, 25%, median, 75%, max, skewness, kurtosis).
  - Computes per-column missingness counts, missing ratios, and overall dataset missingness percentage.
  - Generates Pearson numerical correlation matrix for interaction analysis.

---

## 💻 Code Example

```python
from automl_engine.eda import AutoEDAEngine

eda_engine = AutoEDAEngine()
eda_report = eda_engine.profile_dataframe(df_clean)

# eda_report contains:
# - 'column_classification'
# - 'statistical_summary'
# - 'missingness_summary'
# - 'correlation_matrix'
```
