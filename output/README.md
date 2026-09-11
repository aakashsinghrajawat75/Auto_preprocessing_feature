# 💾 `output` — Exported Tensor Artifacts & Processed Datasets

The `output` directory stores exported processed CSVs, Parquet files, NumPy matrix binary files, and fitted model `.pkl` pickle files.

---

## 📂 Subdirectory Structure

For each processed dataset, a dedicated folder is created under `output/`:

```
output/
├── housing_sample/
│   ├── baseline_processed.csv         # Processed CSV via Baseline Naive EDA Pipeline
│   ├── automl_processed.csv           # Processed CSV via AutoML Regex Sanitization Pipeline
│   ├── cleaned_data.csv               # Exported Cleaned Data CSV
│   ├── cleaned_data.parquet           # Exported Parquet Format
│   ├── v_dense_matrix.npy             # Exported 128-d Vector Matrix (NumPy Array)
│   ├── automl_results.pkl             # Exported Results Pickle Bundle
│   └── automl_pipeline_model.pkl      # Fitted AutoMLPipeline Pickle Model
├── products_sample/
├── logs_sample/
└── test_housing/
```
