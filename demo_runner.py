import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from automl_engine.pipeline import AutoMLPipeline


def run_demo():
    print("=" * 80)
    print(" SIH 2026 AutoML Preprocessing, Feature Engineering & Vectorization Engine")
    print("=" * 80)

    dataset_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "output"

    datasets = [
        ("Real Estate Messy CSV", dataset_dir / "housing_sample.csv", None, "Price_Status"),
        ("E-Commerce Nested JSON", dataset_dir / "products_sample.json", None, "stock_status"),
        ("System Logs XML", dataset_dir / "logs_sample.xml", "log", "attr_level"),
    ]

    pipeline = AutoMLPipeline(target_dim=128, imputation_strategy="knn")

    for name, file_path, record_path, target_col in datasets:
        print(f"\n>>> PROCESSING DATASET: {name} [{file_path.name}]")
        print("-" * 60)

        cleaned_df, v_dense, report = pipeline.fit_transform(file_path, record_path=record_path)

        print("\n1. Ingestion Metadata:")
        print(json.dumps(report["ingestion_metadata"], indent=2))

        print("\n2. Regex Auto-Sanitization Report:")
        print(json.dumps(report["sanitization_report"], indent=2))

        print("\n3. Auto-EDA Classification:")
        print(json.dumps(report["eda_report"]["column_classification"], indent=2))

        print("\n4. Cleaned DataFrame Preview:")
        print(cleaned_df.head())

        print(f"\n5. Dense Vector Matrix V_dense Shape: {v_dense.shape}")
        print(f"Sample V_dense Row [0, :8]: {v_dense[0, :8]}")

        # Run Comparative Benchmarking
        if target_col in cleaned_df.columns:
            print(f"\n6. 🎯 Comparative Benchmarking (Baseline Manual EDA vs. AutoML Engine on target '{target_col}'):")
            try:
                eval_res = pipeline.evaluate_models(target_column=target_col)
                print(f"   Task Type: {eval_res['task_type'].upper()} | Primary Metric: {eval_res['metric_type']}")
                print(f"   Best Baseline Model  : {eval_res['best_baseline_model']['model']} (Score: {eval_res['best_baseline_model']['score']})")
                print(f"   Best AutoML Model    : {eval_res['best_automl_model']['model']} (Score: {eval_res['best_automl_model']['score']})")
                print(f"   Average Delta Gain   : +{eval_res['overall_avg_improvement_pct']}%")
                print("\n   Comparison Summary Table:")
                for row in eval_res["comparison_table"]:
                    b_score = row.get(f"Baseline ({eval_res['metric_type']})", 0.0)
                    a_score = row.get(f"AutoML Engine ({eval_res['metric_type']})", 0.0)
                    delta = row["Delta Improvement (%)"]
                    print(f"     - {row['Model']:<15}: Baseline={b_score:<7.4f} | AutoML={a_score:<7.4f} | Gain={delta:+6.2f}% | {row['Status']}")
            except Exception as e:
                print(f"   Benchmarking error: {e}")

        # Export Tensors & Files
        ds_output = output_dir / file_path.stem
        exported_files = pipeline.export_results(cleaned_df, v_dense, ds_output)
        print("\n7. Exported Tensor Artifacts:")
        for k, v in exported_files.items():
            print(f"  - {k}: {v}")

        print("=" * 80)


if __name__ == "__main__":
    run_demo()
