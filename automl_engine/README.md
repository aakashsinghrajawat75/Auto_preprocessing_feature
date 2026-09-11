#  `automl_engine` Package Overview

The `automl_engine` package is the core Object-Oriented Programming (OOP) foundation of the **SIH 2026 AutoML Engine**. It houses all autonomous components required to ingest, sanitize, profile, impute, vectorize, and evaluate tabular datasets.

---

##  Submodule Architecture

| Submodule | Purpose & Class Responsibilities |
| :--- | :--- |
| **[`ingestion/`](ingestion/)** | Multi-format file ingestion (CSV, Excel, JSON, XML) using Factory pattern (`DataIngestorFactory`). |
| **[`sanitization/`](sanitization/)** | Regex-based numeric token extraction & unit scale normalization (`RegexSanitizer`). |
| **[`eda/`](eda/)** | Autonomous statistical dataset profiling & column classification (`AutoEDAEngine`). |
| **[`imputation/`](imputation/)** | Smart numerical NaN imputation (KNN, Median, Mean) & mode categorical imputation (`StatisticalImputer`, `CategoricalImputer`). |
| **[`vectorization/`](vectorization/)** | Dense vector matrix synthesis ($V_{\text{dense}} \in \mathbb{R}^{N \times 128}$) combining numerical scaling, one-hot, and TF-IDF (`MultiColumnVectorizationEngine`). |
| **[`evaluation/`](evaluation/)** | Model benchmarking across 5 ML algorithms evaluating $R^2$, MSE, MAE, RMSE, Accuracy, F1 (`ModelEvaluator`). |
| **[`pipeline/`](pipeline/)** | Unified pipeline orchestrator (`AutoMLPipeline`). |
| **[`utils/`](utils/)** | Artifact serialization & export (.npy, .parquet, .csv, .pkl) (`TensorExporter`). |

---

##  Python Usage Example

```python
from automl_engine.pipeline import AutoMLPipeline

# Instantiate zero-touch pipeline
pipeline = AutoMLPipeline(target_dim=128, imputation_strategy="knn")

# Run fit_transform on raw data
cleaned_df, v_dense, report = pipeline.fit_transform("data/housing_sample.csv")

# Run comparative model benchmarking
eval_report = pipeline.evaluate_models(target_column="price")
```
