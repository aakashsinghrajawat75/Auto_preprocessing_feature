import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from model_comparator import BaselinePipeline, EnginePipeline, ModelComparator


def create_sample_csv_path(tmp_path):
    p = tmp_path / "test_housing.csv"
    content = (
        "id,area,rooms,price,notes\n"
        "P1,1450 sqft,3 BHK,85 Lakhs,Great view\n"
        "P2,850 sq ft,2 BHK,45 Lakhs,Cozy flat\n"
        "P3,2100 sqft,4 BHK,120 Lakhs,Luxury Villa\n"
        "P4,1100 sqft,2 BHK,60 Lakhs,Modern apartment\n"
        "P5,1600 sqft,3 BHK,95 Lakhs,Garden home\n"
        "P6,750 sqft,1 BHK,35 Lakhs,Studio flat\n"
    )
    p.write_text(content)
    return p


def test_baseline_pipeline(sample_csv_path):
    pipe = BaselinePipeline(target_column="price")
    df_proc, csv_path = pipe.fit_transform(sample_csv_path)
    assert isinstance(df_proc, pd.DataFrame)
    assert csv_path.exists()
    assert "baseline_processed.csv" in csv_path.name


def test_engine_pipeline(sample_csv_path):
    pipe = EnginePipeline(target_dim=128)
    df_proc, v_dense, csv_path = pipe.fit_transform(sample_csv_path)
    assert isinstance(df_proc, pd.DataFrame)
    assert isinstance(v_dense, np.ndarray)
    assert csv_path.exists()
    assert "automl_processed.csv" in csv_path.name


def test_model_comparator(sample_csv_path):
    comparator = ModelComparator(target_dim=128)
    res = comparator.run_comparison(sample_csv_path, target_column="price")
    assert "task_type" in res
    assert res["task_type"] == "regression"
    assert "comparison_table" in res
    assert len(res["comparison_table"]) == 5
    assert Path(res["baseline_csv_path"]).exists()
    assert Path(res["automl_csv_path"]).exists()


if __name__ == "__main__":
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        p = create_sample_csv_path(tmp_path)
        print("Running test_baseline_pipeline...")
        test_baseline_pipeline(p)
        print("Running test_engine_pipeline...")
        test_engine_pipeline(p)
        print("Running test_model_comparator...")
        test_model_comparator(p)
        print("\nALL MODEL COMPARATOR UNIT TESTS PASSED SUCCESSFULLY! (3/3 passed)")
