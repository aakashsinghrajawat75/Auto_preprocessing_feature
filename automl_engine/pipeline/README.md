# 🚀 `automl_engine/pipeline` — AutoML Pipeline Orchestrator

The `pipeline` module provides the unified end-to-end orchestrator class `AutoMLPipeline`.

---

## 📄 File Inventory & Class Responsibilities

### `automl_pipeline.py`
- **`AutoMLPipeline`**:
  - `fit_transform(file_path_or_df)`: Executes ingestion $\rightarrow$ regex sanitization $\rightarrow$ Auto-EDA $\rightarrow$ imputation $\rightarrow$ vectorization $\rightarrow$ return `(cleaned_df, v_dense, report)`.
  - `evaluate_models(target_column)`: Executes model benchmarking comparison.
  - `save_model(filepath)`: Serializes pipeline instance to `.pkl` pickle format.
  - `load_model(filepath)`: Class method loading fitted pipeline from `.pkl` file.

---

## 💻 Code Example

```python
from automl_engine.pipeline import AutoMLPipeline

pipeline = AutoMLPipeline(target_dim=128)
cleaned_df, v_dense, report = pipeline.fit_transform("data/housing_sample.csv")

# Save model pickle
pipeline.save_model("fitted_pipeline.pkl")
```
