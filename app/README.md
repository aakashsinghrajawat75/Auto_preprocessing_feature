# 🖥️ `app` — Streamlit Main Web Application Layer

The `app` folder houses the interactive web application built with Streamlit.

---

##  File Inventory & Application Features

### `app.py`
- **Interactive Controls**:
  - Data Source Selection (Built-in sample datasets, custom file upload, load `.pkl` model).
  - Hyperparameter tuning sliders (Vector Dimension $D$, Imputation Strategy, Scaler Type).
- **Tabbed Dashboard**:
  - **Tab 1:  Ingestion & Data**: Previews ingested raw metadata and cleaned DataFrame.
  - **Tab 2:  Auto-EDA Dashboard**: Visualizes column dtypes, statistical metrics, missingness ratios, and correlation matrix.
  - **Tab 3:  Regex Sanitization**: Displays extracted numeric values and regex sanitization logs.
  - **Tab 4:  NaN Imputation**: Displays missing value imputation summary.
  - **Tab 5:  Dense Vector $V_{\text{dense}}$**: Inspects synthesized 128-d numerical vector embedding matrix.
  - **Tab 6:  Model Benchmarking**: Evaluates 5 ML algorithms (Linear, DT, RF, KNN, XGBoost).
  - **Tab 7:  Pickling (.pkl) & Export**: Direct downloads for `.csv`, `.parquet`, `.npy`, and `.pkl` models.

---

##  How to Run

```bash
streamlit run app/app.py
```
