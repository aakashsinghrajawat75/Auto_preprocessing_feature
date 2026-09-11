import sys
import os
import json
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_comparator.comparator import ModelComparator
from model_comparator.baseline_pipeline import BaselinePipeline
from model_comparator.engine_pipeline import EnginePipeline


# Set Page Configuration
st.set_page_config(
    page_title="SIH 2026 - Dedicated Model Benchmarking Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1F2937;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #6366F1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">⚖️ SIH 2026 Model Comparator & Visualization Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Step-by-Step Baseline CSV vs. SIH AutoML Engine CSV Feature Engineering Benchmarking (R², MSE, MAE, RMSE, Accuracy)</div>', unsafe_allow_html=True)

    # Sidebar Options
    st.sidebar.title("⚙️ Comparator Settings")

    data_mode = st.sidebar.radio(
        "Select Data Source:",
        ["Built-in Sample Datasets", "Upload Custom File"]
    )

    sample_datasets_dir = PROJECT_ROOT / "data"
    input_file_path = None

    if data_mode == "Built-in Sample Datasets":
        sample_choice = st.sidebar.selectbox(
            "Choose Sample Dataset:",
            ["Housing Messy (CSV)", "E-Commerce Products (JSON)", "System Logs (XML)"]
        )
        if sample_choice == "Housing Messy (CSV)":
            input_file_path = sample_datasets_dir / "housing_sample.csv"
            default_target = "Price_Status"
        elif sample_choice == "E-Commerce Products (JSON)":
            input_file_path = sample_datasets_dir / "products_sample.json"
            default_target = "stock_status"
        else:
            input_file_path = sample_datasets_dir / "logs_sample.xml"
            default_target = "attr_level"

    elif data_mode == "Upload Custom File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Dataset File:",
            type=["csv", "tsv", "json", "xml", "xlsx", "xls"]
        )
        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                input_file_path = Path(tmp.name)
            default_target = None

    if input_file_path is None:
        st.info("👈 Please select or upload a dataset file from the sidebar to begin comparison.")
        return

    # Inspect column names for target selection
    from automl_engine.ingestion.factory import DataIngestorFactory
    record_path = "log" if str(input_file_path).endswith(".xml") else None
    ingestor = DataIngestorFactory.get_ingestor(input_file_path)
    preview_df = ingestor.load_data(record_path=record_path) if record_path else ingestor.load_data()

    col_options = list(preview_df.columns)
    default_idx = 0
    if 'default_target' in locals() and default_target in col_options:
        default_idx = col_options.index(default_target)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Target & Tuning Configuration")
    target_column = st.sidebar.selectbox("Target Column to Predict:", col_options, index=default_idx)
    force_task = st.sidebar.selectbox("Task Override (Optional):", ["Auto-Detect", "regression", "classification"])
    task_override = None if force_task == "Auto-Detect" else force_task

    imputation_strategy = st.sidebar.selectbox("Imputation Strategy:", ["knn", "median", "mean"])
    target_dim = st.sidebar.slider("Vector Dimension (V_dense):", 16, 256, 128, step=16)

    run_comp = st.sidebar.button("🚀 Run Dual CSV Generation & Comparison", type="primary", use_container_width=True)

    if run_comp or "comp_results" in st.session_state:
        if run_comp or "comp_results" not in st.session_state:
            with st.spinner("Generating baseline_processed.csv, automl_processed.csv, and benchmarking 5 ML models..."):
                comparator = ModelComparator(
                    target_dim=target_dim,
                    imputation_strategy=imputation_strategy
                )
                res = comparator.run_comparison(
                    file_path_or_df=input_file_path,
                    target_column=target_column,
                    record_path=record_path,
                    task_type=task_override
                )
                st.session_state["comp_results"] = res

        res = st.session_state["comp_results"]

        # Summary Metric Header Bar
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Detected Task", res["task_type"].upper(), f"Primary Metric: {res['metric_type']}")
        with m2:
            st.metric("Best Baseline Model", res["best_baseline_model"]["model"], f"Score: {res['best_baseline_model']['score']}")
        with m3:
            st.metric("Best AutoML Model", res["best_automl_model"]["model"], f"Score: {res['best_automl_model']['score']}")
        with m4:
            st.metric("Average Accuracy Gain (Δ)", f"+{res['overall_avg_improvement_pct']}%")

        st.markdown("---")

        tab_csvs, tab_table, tab_charts, tab_details = st.tabs([
            "📄 Generated CSV Inspection",
            "🎯 Comparative Metric Summary",
            "📊 Interactive R² / MSE / Accuracy Charts",
            "🔬 Detailed Model Breakdown"
        ])

        with tab_csvs:
            st.subheader("Step 1 & 2: Processed CSV Comparisons")
            col_csv1, col_csv2 = st.columns(2)

            with col_csv1:
                st.markdown("#### 1. Baseline Naive EDA (`baseline_processed.csv`)")
                st.caption(f"File Path: `{res['baseline_csv_path']}`")
                st.dataframe(res["baseline_df"].head(15), use_container_width=True)
                st.download_button(
                    label="📄 Download baseline_processed.csv",
                    data=res["baseline_df"].to_csv(index=False),
                    file_name="baseline_processed.csv",
                    mime="text/csv"
                )

            with col_csv2:
                st.markdown("#### 2. SIH AutoML Engine (`automl_processed.csv`)")
                st.caption(f"File Path: `{res['automl_csv_path']}`")
                st.dataframe(res["automl_df"].head(15), use_container_width=True)
                st.download_button(
                    label="⚡ Download automl_processed.csv",
                    data=res["automl_df"].to_csv(index=False),
                    file_name="automl_processed.csv",
                    mime="text/csv"
                )

        with tab_table:
            st.subheader("Comparative ML Model Benchmarking Summary Table")
            st.markdown(f"Evaluates **5 ML Algorithms** on target variable **'{res['target_column']}'**:")
            comp_df = pd.DataFrame(res["comparison_table"])
            st.dataframe(comp_df, use_container_width=True)

        with tab_charts:
            st.subheader(f"📊 Performance Comparison Chart ({res['metric_type']})")
            comp_df = pd.DataFrame(res["comparison_table"])
            b_col_name = [c for c in comp_df.columns if "Baseline" in c][0]
            a_col_name = [c for c in comp_df.columns if "AutoML Engine" in c][0]

            fig, ax = plt.subplots(figsize=(10, 4.5))
            models_list = comp_df["Model"]
            x = np.arange(len(models_list))
            width = 0.35

            ax.bar(x - width/2, comp_df[b_col_name], width, label="Baseline Manual EDA CSV", color="#6B7280")
            ax.bar(x + width/2, comp_df[a_col_name], width, label="SIH AutoML Engine CSV", color="#6366F1")

            ax.set_ylabel(res['metric_type'], fontsize=12)
            ax.set_title(f"Model Performance Comparison: Baseline CSV vs AutoML CSV ({res['metric_type']})", fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(models_list, fontsize=10)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.5)

            st.pyplot(fig)

            # Extra metrics for Regression (MSE, MAE, RMSE)
            if res["task_type"] == "regression":
                st.markdown("#### 📉 Regression Error Metrics Comparison (Lower is Better)")
                reg_table = []
                for m in res["baseline_metrics"].keys():
                    b_m = res["baseline_metrics"][m]
                    a_m = res["automl_metrics"][m]
                    reg_table.append({
                        "Model": m,
                        "Baseline RMSE": b_m.get("RMSE", 0.0),
                        "AutoML RMSE": a_m.get("RMSE", 0.0),
                        "Baseline MAE": b_m.get("MAE", 0.0),
                        "AutoML MAE": a_m.get("MAE", 0.0),
                        "Baseline MSE": b_m.get("MSE", 0.0),
                        "AutoML MSE": a_m.get("MSE", 0.0),
                    })
                st.dataframe(pd.DataFrame(reg_table), use_container_width=True)

        with tab_details:
            st.subheader("🔬 Detailed Metric Breakdown per Algorithm")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("#### Baseline Pipeline Raw Metrics")
                st.json(res["baseline_metrics"])
            with col_d2:
                st.markdown("#### AutoML Engine Pipeline Raw Metrics")
                st.json(res["automl_metrics"])


if __name__ == "__main__":
    main()
