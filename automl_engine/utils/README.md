# 💾 `automl_engine/utils` — Artifact Exporters & Pickle Handlers

The `utils` module handles exporting preprocessed datasets, dense matrices, and pickle models to disk.

---

## 📄 File Inventory & Class Responsibilities

### `exporters.py`
- **`TensorExporter`**:
  - `export_all(df_cleaned, v_dense, eda_report, output_dir)`: Saves `.csv`, `.parquet`, `.npy` vector matrix, `.pkl` dataset bundle, and `.pkl` pipeline model.
  - `save_pickle(obj, filepath)`: Serializes Python objects to `.pkl`.
  - `load_pickle(filepath)`: Deserializes `.pkl` objects from disk.

---

## 💻 Code Example

```python
from automl_engine.utils import TensorExporter

exported_files = TensorExporter.export_all(cleaned_df, v_dense, eda_report, "output/housing")
```
