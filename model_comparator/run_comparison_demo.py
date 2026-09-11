import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_comparator.comparator import ModelComparator


def run_demo():
    print("=" * 80)
    print(" SIH 2026 Dedicated Model Comparator Demo (Baseline CSV vs. AutoML Engine CSV)")
    print("=" * 80)

    dataset_dir = PROJECT_ROOT / "data"

    datasets = [
        ("Real Estate Messy CSV", dataset_dir / "housing_sample.csv", None, "Price_Status"),
        ("E-Commerce Nested JSON", dataset_dir / "products_sample.json", None, "stock_status"),
        ("System Logs XML", dataset_dir / "logs_sample.xml", "log", "attr_level"),
    ]

    comparator = ModelComparator(target_dim=128, imputation_strategy="knn")

    for name, file_path, record_path, target_col in datasets:
        print(f"\n>>> EXECUTING MODEL COMPARATOR ON DATASET: {name} [{file_path.name}]")
        print("-" * 60)

        res = comparator.run_comparison(
            file_path_or_df=file_path,
            target_column=target_col,
            record_path=record_path
        )

        print(f"1. Generated Baseline CSV : {res['baseline_csv_path']}")
        print(f"2. Generated AutoML CSV   : {res['automl_csv_path']}")
        print(f"3. Detected Task Type     : {res['task_type'].upper()}")
        print(f"4. Primary Metric         : {res['metric_type']}")
        print(f"5. Best Baseline Model    : {res['best_baseline_model']['model']} ({res['best_baseline_model']['score']})")
        print(f"6. Best AutoML Engine     : {res['best_automl_model']['model']} ({res['best_automl_model']['score']})")
        print(f"7. Average Metric Gain    : +{res['overall_avg_improvement_pct']}%")

        print("\n8. Step-by-Step Model Metric Comparison Summary:")
        print("   " + "-" * 70)
        for row in res["comparison_table"]:
            b_score = row.get(f"Baseline ({res['metric_type']})", 0.0)
            a_score = row.get(f"AutoML Engine ({res['metric_type']})", 0.0)
            delta = row["Delta Improvement (%)"]
            print(f"   - {row['Model']:<15}: Baseline={b_score:<7.4f} | AutoML={a_score:<7.4f} | Gain={delta:+6.2f}% | {row['Status']}")

        print("=" * 80)


if __name__ == "__main__":
    run_demo()
