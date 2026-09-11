from pathlib import Path
from typing import Union, Type
from automl_engine.ingestion.base import BaseDataIngestor
from automl_engine.ingestion.csv_ingestor import CSVIngestor
from automl_engine.ingestion.excel_ingestor import ExcelIngestor
from automl_engine.ingestion.json_ingestor import JSONIngestor
from automl_engine.ingestion.xml_ingestor import XMLIngestor


class DataIngestorFactory:
    """
    Factory Pattern for instantiating the appropriate DataIngestor based on file extension.
    """

    _INGESTOR_MAP = {
        ".csv": CSVIngestor,
        ".tsv": CSVIngestor,
        ".txt": CSVIngestor,
        ".xlsx": ExcelIngestor,
        ".xls": ExcelIngestor,
        ".json": JSONIngestor,
        ".jsonl": JSONIngestor,
        ".xml": XMLIngestor,
    }

    @classmethod
    def get_ingestor(cls, file_path: Union[str, Path]) -> BaseDataIngestor:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in cls._INGESTOR_MAP:
            raise ValueError(
                f"Unsupported file extension '{ext}'. Supported formats: {list(cls._INGESTOR_MAP.keys())}"
            )

        ingestor_cls: Type[BaseDataIngestor] = cls._INGESTOR_MAP[ext]
        return ingestor_cls(path)
