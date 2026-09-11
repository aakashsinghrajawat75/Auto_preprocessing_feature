# 🎯 `automl_engine/evaluation` — Model Benchmarking & Evaluator Layer

The `evaluation` module benchmarks predictive performance across machine learning algorithms evaluating metrics ($R^2$, MSE, MAE, RMSE, Accuracy, F1-Score).

---

## 📄 File Inventory & Class Responsibilities

### `model_evaluator.py`
- **`ModelEvaluator`**:
  - **Task Autodetection**: Automatically detects whether target variable is **Regression** or **Classification**.
  - **5 ML Algorithm Families**:
    - Linear Models (Linear Regression / Logistic Regression with `class_weight='balanced'`)
    - Decision Trees (`DecisionTreeRegressor` / `DecisionTreeClassifier`)
    - Random Forests (`RandomForestRegressor` / `RandomForestClassifier`)
    - K-Nearest Neighbors (`KNeighborsRegressor` / `KNeighborsClassifier`)
    - XGBoost (`XGBRegressor` / `XGBClassifier` with Gradient Boosting fallback)
  - **Metrics Computed**:
    - Regression: $R^2$, MSE, MAE, RMSE.
    - Classification: Accuracy, Precision, Recall, F1-Score.

---

## 💻 Code Example

```python
from automl_engine.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
report = evaluator.compare_pipelines(raw_df, cleaned_df, target_col="price", v_dense=v_dense)
```
