import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from automl_engine.ingestion import DataIngestorFactory, CSVIngestor, JSONIngestor, XMLIngestor
from automl_engine.sanitization import RegexSanitizer
from automl_engine.eda import AutoEDAEngine
from automl_engine.imputation import StatisticalImputer, CategoricalImputer
from automl_engine.vectorization import MultiColumnVectorizationEngine
from automl_engine.evaluation import ModelEvaluator
from automl_engine.pipeline import AutoMLPipeline


def create_sample_csv_path(tmp_path):
    p = tmp_path / "test_data.csv"
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


def test_csv_ingestor(sample_csv_path):
    ingestor = DataIngestorFactory.get_ingestor(sample_csv_path)
    df = ingestor.load_data()
    assert len(df) == 6
    assert "area" in df.columns


def test_regex_sanitizer():
    sanitizer = RegexSanitizer()
    df = pd.DataFrame({
        "area": ["1,450 sq.ft.", "850 sq ft", None],
        "bhk": ["3 BHK", "2 BHK", "4 BHK"],
        "price": ["85 Lakhs", "45 Lakhs", None],
        "desc": ["Very long description sentence about luxury housing with pool", "Short description", "Another sentence"]
    })
    df_clean, report = sanitizer.sanitize_dataframe(df)

    assert df_clean["area"].dtype == "float64"
    assert df_clean["area"].iloc[0] == 1450.0
    assert df_clean["bhk"].iloc[0] == 3.0
    assert df_clean["price"].iloc[0] == 8500000.0
    assert "area" in report


def test_imputers():
    df = pd.DataFrame({
        "num1": [10.0, 20.0, np.nan, 40.0, 50.0],
        "num2": [1.0, np.nan, 3.0, 4.0, 5.0],
        "cat1": ["A", "A", np.nan, "B", "A"]
    })
    num_imp = StatisticalImputer(strategy="knn", n_neighbors=2)
    df_num = num_imp.fit_transform(df)
    assert not df_num["num1"].isna().any()
    assert not df_num["num2"].isna().any()

    cat_imp = CategoricalImputer(strategy="most_frequent")
    df_cat = cat_imp.fit_transform(df)
    assert not df_cat["cat1"].isna().any()
    assert df_cat["cat1"].iloc[2] == "A"


def test_pipeline_end_to_end(sample_csv_path):
    pipeline = AutoMLPipeline(target_dim=128)
    df_clean, v_dense, report = pipeline.fit_transform(sample_csv_path)

    assert isinstance(df_clean, pd.DataFrame)
    assert isinstance(v_dense, np.ndarray)
    assert v_dense.shape == (6, 128)
    assert "ingestion_metadata" in report
    assert "eda_report" in report


def test_model_evaluator(sample_csv_path):
    pipeline = AutoMLPipeline(target_dim=128)
    df_clean, v_dense, _ = pipeline.fit_transform(sample_csv_path)

    eval_res = pipeline.evaluate_models(target_column="price")
    assert "task_type" in eval_res
    assert eval_res["task_type"] == "regression"
    assert "comparison_table" in eval_res
    assert len(eval_res["comparison_table"]) == 5
    assert "best_automl_model" in eval_res


def test_pickle_serialization(sample_csv_path, tmp_path):
    pipeline = AutoMLPipeline(target_dim=128)
    df_clean, v_dense, report = pipeline.fit_transform(sample_csv_path)

    pkl_file = tmp_path / "model.pkl"
    saved_path = pipeline.save_model(pkl_file)
    assert Path(saved_path).exists()

    loaded_pipeline = AutoMLPipeline.load_model(pkl_file)
    assert isinstance(loaded_pipeline, AutoMLPipeline)
    assert loaded_pipeline.target_dim == 128


if __name__ == "__main__":
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        p = create_sample_csv_path(tmp_path)
        print("Running test_csv_ingestor...")
        test_csv_ingestor(p)
        print("Running test_regex_sanitizer...")
        test_regex_sanitizer()
        print("Running test_imputers...")
        test_imputers()
        print("Running test_pipeline_end_to_end...")
        test_pipeline_end_to_end(p)
        print("Running test_model_evaluator...")
        test_model_evaluator(p)
        print("Running test_pickle_serialization...")
        test_pickle_serialization(p, tmp_path)
        print("\nALL UNIT TESTS PASSED SUCCESSFULLY! (6/6 passed)")
