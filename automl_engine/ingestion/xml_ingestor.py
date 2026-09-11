import pandas as pd
import xml.etree.ElementTree as ET
from typing import Dict, List, Any
from pathlib import Path
from automl_engine.ingestion.base import BaseDataIngestor


class XMLIngestor(BaseDataIngestor):
    """
    Concrete Data Ingestor for XML files.
    Parses XML nodes, attributes, and text values into a clean tabular DataFrame.
    """

    def load_data(self, record_tag: str = None, **kwargs) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"XML file not found at path: {self.file_path}")

        tree = ET.parse(self.file_path)
        root = tree.getroot()

        records: List[Dict[str, Any]] = []

        # Determine target nodes to iterate over
        if record_tag:
            nodes = root.findall(f".//{record_tag}")
        else:
            # By default, use children of root node
            nodes = list(root)
            if not nodes:
                nodes = [root]

        for node in nodes:
            row: Dict[str, Any] = {}
            # Include attributes of node
            for attr_k, attr_v in node.attrib.items():
                row[f"attr_{attr_k}"] = attr_v

            # Include direct child text tags
            for child in node:
                child_tag = child.tag
                child_text = child.text.strip() if child.text else None
                row[child_tag] = child_text

            # Include direct text if leaf node
            if not list(node) and node.text and node.text.strip():
                row[node.tag] = node.text.strip()

            if row:
                records.append(row)

        df = pd.DataFrame(records)
        self._update_metadata(df)
        return df
