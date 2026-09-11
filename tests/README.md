# 🧪 `tests` — Automated Unit Test Suite

The `tests` directory contains automated test suites verifying core AutoML engine components, data ingestors, imputers, vectorizers, and model benchmarking.

---

## 📄 File Inventory & Test Coverage

### 1. `test_automl_engine.py` (6/6 Passed)
- `test_csv_ingestor()`: Validates CSV delimiter inference and loading.
- `test_regex_sanitizer()`: Validates numeric token extraction and unit conversions (`"85 Lakhs" -> 8500000.0`).
- `test_imputers()`: Validates KNN numeric imputation and mode categorical imputation.
- `test_pipeline_end_to_end()`: Validates end-to-end `AutoMLPipeline.fit_transform()`.
- `test_model_evaluator()`: Validates `ModelEvaluator` task detection and metrics.
- `test_pickle_serialization()`: Validates pipeline `.pkl` save and load.

### 2. `test_model_comparator.py` (3/3 Passed)
- `test_baseline_pipeline()`: Validates baseline CSV creation (`baseline_processed.csv`).
- `test_engine_pipeline()`: Validates AutoML CSV creation (`automl_processed.csv`).
- `test_model_comparator()`: Validates dual pathway model comparison.

---

## 🚀 Execution Command

```bash
python3 tests/test_automl_engine.py
python3 tests/test_model_comparator.py
```
