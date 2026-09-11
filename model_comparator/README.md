#  `model_comparator` — Dedicated Model Benchmarking Package

The `model_comparator` subfolder isolates the dual-pathway feature generation and comparative machine learning evaluation.

---

##  File Inventory & Class Responsibilities

### 1. `baseline_pipeline.py`
- **`BaselinePipeline`**: Simulates naive manual EDA (median fill, basic label encoding without regex string unit extraction) and exports `output/<dataset>/baseline_processed.csv`.

### 2. `engine_pipeline.py`
- **`EnginePipeline`**: Executes zero-touch AutoML engine pipeline (regex unit extraction `"1,450 sq.ft." -> 1450.0`, KNN imputation, 128-d vector matrix) and exports `output/<dataset>/automl_processed.csv`.

### 3. `comparator.py`
- **`ModelComparator`**: Evaluates 5 ML algorithms (**Linear/Logistic Regression, Decision Tree, Random Forest, KNN, XGBoost**) across both CSVs, calculating $R^2$, MSE, MAE, RMSE, Accuracy, Precision, Recall, F1-Score, and percentage deltas ($\Delta_{\%}$).

### 4. `app.py`
- Dedicated Streamlit Visualization App providing side-by-side CSV previews, comparison tables, and performance charts.

### 5. `run_comparison_demo.py`
- Standalone CLI execution script.

---

##  How to Run

```bash
# Launch Dedicated Streamlit Dashboard
streamlit run model_comparator/app.py

# Run CLI Demo Script
python3 model_comparator/run_comparison_demo.py
```
