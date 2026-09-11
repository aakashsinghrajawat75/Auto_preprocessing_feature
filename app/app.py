import sys
import os
import json
import tempfile
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from automl_engine.pipeline import AutoMLPipeline
from automl_engine.utils import TensorExporter


# Set Page Configuration
st.set_page_config(
    page_title="SIH 2026 - AutoML Preprocessing & Vectorization Engine",
    page_icon="⚡",
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 6px;
        padding: 0px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">⚡ SIH 2026 AutoML Preprocessing & Vectorization Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Zero-Touch Ingestion (CSV, JSON, XML, Excel) • Regex Auto-Sanitization • Auto-EDA • Smart NaN Imputation • 128-d Dense Tensor Synthesis</div>', unsafe_allow_html=True)

    # Sidebar Options
    st.sidebar.title("⚙️ Engine Configuration")

    data_mode = st.sidebar.radio(
        "Select Data Source:",
        ["Built-in Sample Datasets", "Upload Custom File", "Load Pre-trained .pkl Model"]
    )

    sample_datasets_dir = PROJECT_ROOT / "data"

    pipeline = None
    input_file_path = None
    df_raw = None

    if data_mode == "Built-in Sample Datasets":
        sample_choice = st.sidebar.selectbox(
            "Choose Sample Dataset:",
            ["Housing Messy (CSV)", "E-Commerce Products (JSON)", "System Logs (XML)"]
        )
        if sample_choice == "Housing Messy (CSV)":
            input_file_path = sample_datasets_dir / "housing_sample.csv"
        elif sample_choice == "E-Commerce Products (JSON)":
            input_file_path = sample_datasets_dir / "products_sample.json"
        else:
            input_file_path = sample_datasets_dir / "logs_sample.xml"

    elif data_mode == "Upload Custom File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Dataset File:",
            type=["csv", "tsv", "json", "xml", "xlsx", "xls"]
        )
        if uploaded_file is not None:
            # Save uploaded file to temp file
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                input_file_path = Path(tmp.name)

    elif data_mode == "Load Pre-trained .pkl Model":
        uploaded_pkl = st.sidebar.file_uploader("Upload .pkl Model File:", type=["pkl"])
        if uploaded_pkl is not None:
            try:
                pipeline = AutoMLPipeline.load_model(uploaded_pkl)
                st.sidebar.success("Successfully loaded pre-trained AutoMLPipeline from .pkl!")
            except Exception as e:
                st.sidebar.error(f"Failed to load .pkl model: {e}")

    # Pipeline Tuning Parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Pipeline Hyperparameters")
    target_dim = st.sidebar.slider("Target Dense Vector Dimension (V_dense):", 16, 256, 128, step=16)
    imputation_strategy = st.sidebar.selectbox("Imputation Strategy:", ["knn", "median", "mean", "iterative"])
    scaler_type = st.sidebar.selectbox("Scaler Type:", ["standard", "minmax", "robust"])

    run_pipeline = st.sidebar.button("🚀 Run Zero-Touch AutoML Pipeline", type="primary", use_container_width=True)

    if input_file_path is not None and (run_pipeline or "last_results" not in st.session_state):
        with st.spinner("Processing zero-touch AutoML pipeline..."):
            pipeline = AutoMLPipeline(
                target_dim=target_dim,
                imputation_strategy=imputation_strategy,
                scaler_type=scaler_type
            )

            record_path = "log" if str(input_file_path).endswith(".xml") else None
            df_cleaned, v_dense, report = pipeline.fit_transform(input_file_path, record_path=record_path)

            st.session_state["last_results"] = {
                "cleaned_df": df_cleaned,
                "v_dense": v_dense,
                "report": report,
                "pipeline": pipeline,
                "file_name": input_file_path.name
            }

    if "last_results" in st.session_state:
        res = st.session_state["last_results"]
        df_cleaned = res["cleaned_df"]
        v_dense = res["v_dense"]
        report = res["report"]
        pipe = res["pipeline"]

        # Top Summary Metric Bar
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Rows", report["ingestion_metadata"].get("loaded_rows", len(df_cleaned)))
        with col2:
            st.metric("Total Columns", report["ingestion_metadata"].get("loaded_columns", len(df_cleaned.columns)))
        with col3:
            st.metric("Sanitized Columns", len(report["sanitization_report"]))
        with col4:
            st.metric("Missing Cell %", f"{report['eda_report']['dataset_overview']['overall_missing_percentage']}%")
        with col5:
            st.metric("Dense Vector Shape", f"{v_dense.shape[0]} x {v_dense.shape[1]}")

        st.markdown("---")

        # Tabs Navigation
        tab_ingest, tab_eda, tab_sanitization, tab_imputation, tab_vector, tab_benchmark, tab_download = st.tabs([
            "📁 Ingestion & Data",
            "📊 Auto-EDA Dashboard",
            "🧹 Regex Sanitization",
            "🩹 NaN Imputation",
            "🧬 Dense Vector V_dense",
            "🎯 Model Benchmarking (Baseline vs AutoML)",
            "💾 Model Pickling (.pkl) & Export"
        ])

        with tab_ingest:
            st.subheader("Raw Ingestion Metadata & Data Preview")
            st.json(report["ingestion_metadata"])
            st.dataframe(df_cleaned.head(20), use_container_width=True)

        with tab_eda:
            st.subheader("Autonomous Dataset Profiling & Statistics")
            st.markdown("#### Column Classification")
            st.json(report["eda_report"]["column_classification"])

            st.markdown("#### Statistical Summary")
            st.dataframe(pd.DataFrame(report["eda_report"]["statistical_summary"]).T, use_container_width=True)

            if report["eda_report"]["correlation_matrix"]:
                st.markdown("#### Numerical Correlation Matrix")
                st.dataframe(pd.DataFrame(report["eda_report"]["correlation_matrix"]), use_container_width=True)

        with tab_sanitization:
            st.subheader("Regex-Based Token Extraction Report")
            st.info("Extracts numbers from unit strings like '1,450 sq.ft.' -> 1450.0, '3 BHK' -> 3.0, '85 Lakhs' -> 85.0")
            if report["sanitization_report"]:
                st.json(report["sanitization_report"])
            else:
                st.write("No string-to-numeric conversions were required for this dataset.")

        with tab_imputation:
            st.subheader("Missing Value Imputation Overview")
            st.write(f"Numerical Strategy: `{imputation_strategy}` | Categorical Strategy: `most_frequent`")
            missing_df = pd.DataFrame(report["eda_report"]["missingness_summary"]).T
            st.dataframe(missing_df, use_container_width=True)

        with tab_vector:
            st.subheader("Synthesized Dense Vector Matrix ($V_{\\text{dense}}$)")
            st.write(f"Target Vector Dimension $D$: `{v_dense.shape[1]}` | Total Samples: `{v_dense.shape[0]}`")
            st.dataframe(pd.DataFrame(v_dense), use_container_width=True)

        with tab_benchmark:
            st.subheader("🎯 Comparative Benchmarking: Baseline Manual EDA vs. SIH AutoML Engine")
            st.markdown(
                "Compare predictive accuracy and metric performance across **5 ML Model Families** "
                "(**Linear Model, Decision Tree, Random Forest, KNN, XGBoost**) evaluating standard naive manual EDA vs. "
                "the SIH AutoML Preprocessing & Vectorization Engine."
            )

            target_options = list(df_cleaned.columns)
            default_idx = 0
            # Prefer numeric or price/target named column if exists
            for i, c in enumerate(target_options):
                if c.lower() in ["price", "target", "label", "cost", "salary"]:
                    default_idx = i
                    break

            col_select1, col_select2 = st.columns([3, 1])
            with col_select1:
                target_col = st.selectbox("Select Target Variable to Predict:", target_options, index=default_idx)
            with col_select2:
                force_task = st.selectbox("Task Override (Optional):", ["Auto-Detect", "regression", "classification"])
                task_override = None if force_task == "Auto-Detect" else force_task

            if st.button("🚀 Run Comparative Model Benchmarking", type="primary", use_container_width=True):
                with st.spinner(f"Training and benchmarking 5 ML models on '{target_col}'..."):
                    try:
                        eval_res = pipe.evaluate_models(
                            target_column=target_col,
                            task_type=task_override
                        )

                        st.success(f"Benchmarking complete! Detected task: **{eval_res['task_type'].upper()}** | Metric: **{eval_res['metric_type']}**")

                        # Summary Metric Cards
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        with mc1:
                            st.metric(
                                label="Best Baseline Model",
                                value=eval_res["best_baseline_model"]["model"],
                                delta=f"{eval_res['best_baseline_model']['score']} {eval_res['metric_type']}"
                            )
                        with mc2:
                            st.metric(
                                label="Best AutoML Engine Model",
                                value=eval_res["best_automl_model"]["model"],
                                delta=f"{eval_res['best_automl_model']['score']} {eval_res['metric_type']}"
                            )
                        with mc3:
                            st.metric(
                                label="Average Accuracy Gain (Δ)",
                                value=f"+{eval_res['overall_avg_improvement_pct']}%",
                                delta_color="normal"
                            )
                        with mc4:
                            st.metric(
                                label="Evaluated Algorithms",
                                value="5 Models",
                                delta="Linear, DT, RF, KNN, XGBoost"
                            )

                        st.markdown("---")
                        st.markdown("### 📊 Side-by-Side Comparison Table")
                        comp_df = pd.DataFrame(eval_res["comparison_table"])
                        st.dataframe(comp_df, use_container_width=True)

                        # Render Comparison Bar Chart
                        st.markdown("### 📈 Metric Performance Chart (Baseline vs AutoML Engine)")
                        fig, ax = plt.subplots(figsize=(10, 4.5))
                        models_list = comp_df["Model"]
                        b_col_name = [c for c in comp_df.columns if "Baseline" in c][0]
                        a_col_name = [c for c in comp_df.columns if "AutoML Engine" in c][0]

                        x = np.arange(len(models_list))
                        width = 0.35

                        ax.bar(x - width/2, comp_df[b_col_name], width, label="Baseline Manual EDA", color="#6B7280")
                        ax.bar(x + width/2, comp_df[a_col_name], width, label="SIH AutoML Engine", color="#6366F1")

                        ax.set_ylabel(eval_res['metric_type'], fontsize=12)
                        ax.set_title(f"Model Performance Comparison on Target '{target_col}' ({eval_res['metric_type']})", fontsize=14, fontweight="bold")
                        ax.set_xticks(x)
                        ax.set_xticklabels(models_list, fontsize=10)
                        ax.legend()
                        ax.grid(axis='y', linestyle='--', alpha=0.5)

                        st.pyplot(fig)

                        # Detailed Metrics Breakdown
                        st.markdown("### 🔬 Detailed Metrics Breakdown")
                        col_dt1, col_dt2 = st.columns(2)
                        with col_dt1:
                            st.markdown("#### Baseline Naive EDA Metrics")
                            st.json(eval_res["baseline_metrics"])
                        with col_dt2:
                            st.markdown("#### SIH AutoML Engine Metrics")
                            st.json(eval_res["automl_metrics"])

                    except Exception as err:
                        st.error(f"Error executing benchmarking: {err}")

        with tab_download:
            st.subheader("💾 Model Pickling (.pkl) & Artifact Downloads")
            st.markdown("Download pre-trained pipeline models and exported matrices directly to your local workspace:")

            # Create in-memory bytes for Streamlit download buttons
            with tempfile.TemporaryDirectory() as tmp_dir:
                export_paths = pipe.export_results(df_cleaned, v_dense, tmp_dir)

                col_d1, col_d2, col_d3 = st.columns(3)

                with col_d1:
                    if "pkl_pipeline_model" in export_paths:
                        with open(export_paths["pkl_pipeline_model"], "rb") as f:
                            st.download_button(
                                label="📦 Download Model (.pkl)",
                                data=f.read(),
                                file_name="automl_pipeline_model.pkl",
                                mime="application/octet-stream"
                            )

                    with open(export_paths["csv"], "rb") as f:
                        st.download_button(
                            label="📄 Download Cleaned Data (.csv)",
                            data=f.read(),
                            file_name="cleaned_data.csv",
                            mime="text/csv"
                        )

                with col_d2:
                    if "pkl_data" in export_paths:
                        with open(export_paths["pkl_data"], "rb") as f:
                            st.download_button(
                                label="📦 Download Results Bundle (.pkl)",
                                data=f.read(),
                                file_name="automl_results.pkl",
                                mime="application/octet-stream"
                            )

                    with open(export_paths["numpy_matrix"], "rb") as f:
                        st.download_button(
                            label="🧮 Download Vector Matrix (.npy)",
                            data=f.read(),
                            file_name="v_dense_matrix.npy",
                            mime="application/octet-stream"
                        )

                with col_d3:
                    if "parquet" in export_paths:
                        with open(export_paths["parquet"], "rb") as f:
                            st.download_button(
                                label="📈 Download Parquet File (.parquet)",
                                data=f.read(),
                                file_name="cleaned_data.parquet",
                                mime="application/octet-stream"
                            )


if __name__ == "__main__":
    main()
