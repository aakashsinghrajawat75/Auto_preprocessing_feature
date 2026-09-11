# 📥 `automl_engine/ingestion` — Multi-Format Data Ingestion Layer

The `ingestion` layer provides zero-touch loading for flat and complex hierarchical files (**CSV, TSV, Excel, JSON, XML**) without requiring manual schema definitions.

---

## 📄 File Inventory & Class Responsibilities

### 1. `base.py`
- **`BaseIngestor`**: Abstract Base Class enforcing implementation of:
  - `load_data(record_path=None) -> pd.DataFrame`
  - `get_metadata() -> Dict[str, Any]`

### 2. `csv_ingestor.py`
- **`CSVIngestor`**:
  - Automatically infers delimiters (commas `,`, tabs `\t`, semicolons `;`).
  - Autodetects file encoding (`utf-8`, `latin-1`).
  - Standardizes null string tokens (`'N/A'`, `'null'`, `'-'`, `''`) into `np.nan`.

### 3. `excel_ingestor.py`
- **`ExcelIngestor`**: Reads `.xlsx` and `.xls` workbooks with dynamic sheet selection.

### 4. `json_ingestor.py`
- **`JSONIngestor`**: Flattens nested JSON objects into flat DataFrames using `pd.json_normalize`.

### 5. `xml_ingestor.py`
- **`XMLIngestor`**: Parses XML log trees into structured tabular DataFrames.

### 6. `factory.py`
- **`DataIngestorFactory`**:
  - `get_ingestor(file_path)`: Inspects file extension and dynamically instantiates the correct ingestor class.

---

## 💻 Code Example

```python
from automl_engine.ingestion.factory import DataIngestorFactory

# Factory automatically picks CSVIngestor, JSONIngestor, or XMLIngestor
ingestor = DataIngestorFactory.get_ingestor("data/housing_sample.csv")
raw_df = ingestor.load_data()
metadata = ingestor.get_metadata()
```
