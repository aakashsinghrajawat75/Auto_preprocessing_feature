import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union


class TensorExporter:
    """
    Tensor, Data & Model Serialization Exporter.
    Exports clean DataFrames, dense numerical vector matrices, and fitted pipeline models
    to Pickle (.pkl), NumPy (.npy), Parquet, CSV, and PyTorch Tensor formats.
    """

    @staticmethod
    def save_pickle(obj: Any, filepath: Union[str, Path]) -> str:
        """
        Serializes any Python object (e.g. fitted AutoMLPipeline) to a .pkl file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return str(path)

    @staticmethod
    def load_pickle(filepath: Union[str, Path]) -> Any:
        """
        Deserializes a Python object from a .pkl file.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Pickle file not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def export_all(
        df_cleaned: pd.DataFrame,
        v_dense: np.ndarray,
        eda_report: Dict[str, Any],
        output_dir: Union[str, Path],
        pipeline_model: Any = None
    ) -> Dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # 1. Export Cleaned Tabular DataFrame to CSV
        csv_path = output_path / "cleaned_data.csv"
        df_cleaned.to_csv(csv_path, index=False)
        exported_files["csv"] = str(csv_path)

        # 2. Export Cleaned DataFrame to Parquet
        try:
            parquet_path = output_path / "cleaned_data.parquet"
            df_cleaned.to_parquet(parquet_path, index=False)
            exported_files["parquet"] = str(parquet_path)
        except Exception:
            pass

        # 3. Export Dense Vector Matrix V_dense to NumPy (.npy)
        npy_path = output_path / "v_dense_matrix.npy"
        np.save(npy_path, v_dense)
        exported_files["numpy_matrix"] = str(npy_path)

        # 4. Export Object/Data Bundle to Pickle (.pkl)
        pkl_data_path = output_path / "automl_results.pkl"
        results_bundle = {
            "cleaned_df": df_cleaned,
            "v_dense": v_dense,
            "eda_report": eda_report
        }
        TensorExporter.save_pickle(results_bundle, pkl_data_path)
        exported_files["pkl_data"] = str(pkl_data_path)

        # 5. Export Fitted Pipeline Model to Pickle (.pkl) if provided
        if pipeline_model is not None:
            pkl_model_path = output_path / "automl_pipeline_model.pkl"
            TensorExporter.save_pickle(pipeline_model, pkl_model_path)
            exported_files["pkl_pipeline_model"] = str(pkl_model_path)

        # 6. Attempt Export to PyTorch Tensor (.pt)
        try:
            import torch
            pt_path = output_path / "v_dense_tensor.pt"
            tensor = torch.from_numpy(v_dense).float()
            torch.save(tensor, pt_path)
            exported_files["pytorch_tensor"] = str(pt_path)
        except ImportError:
            pass

        return exported_files
