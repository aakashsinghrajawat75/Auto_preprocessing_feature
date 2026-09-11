import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


class RegexSanitizer:
    """
    Auto-Sanitization Engine.
    Uses regex patterns to extract raw floats/integers from messy real-world text strings.
    Example conversions:
      - '1,450 sq.ft.' -> 1450.0
      - '3 BHK' -> 3.0
      - '■ 85 Lakhs' -> 85.0
      - '$ 45,000' -> 45000.0
    """

    def __init__(self, numeric_threshold: float = 0.5):
        """
        :param numeric_threshold: Fraction of non-null values in a text column that must contain
                                  extractable numeric tokens to trigger auto-conversion to float.
        """
        self.numeric_threshold = numeric_threshold
        self.sanitized_columns_info: Dict[str, str] = {}

    def sanitize_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Iterates over string/object columns in DataFrame, extracts embedded numeric values,
        and converts candidates to numerical float types.
        """
        df_clean = df.copy()
        sanitization_report = {}

        for col in df_clean.columns:
            if df_clean[col].dtype == "object" or pd.api.types.is_string_dtype(df_clean[col]):
                series = df_clean[col]
                extracted_series, conversion_ratio = self._attempt_numeric_extraction(series)

                if conversion_ratio >= self.numeric_threshold:
                    df_clean[col] = extracted_series
                    sanitization_report[col] = {
                        "action": "extracted_numeric_float",
                        "conversion_ratio": round(conversion_ratio, 4),
                        "original_dtype": "object",
                        "new_dtype": str(df_clean[col].dtype)
                    }

        return df_clean, sanitization_report

    def _attempt_numeric_extraction(self, series: pd.Series) -> Tuple[pd.Series, float]:
        """
        Extracts numbers from a pandas Series of strings using regex pattern parsing.
        Handles commas in numbers (e.g. 1,450 -> 1450) and currency/unit strings.
        """
        non_nulls = series.dropna()
        if len(non_nulls) == 0:
            return series, 0.0

        extracted_values = []
        successful_extracts = 0

        # Pattern matches numbers with optional commas and decimals
        # e.g. 1,450.50 or 85 or 3.0
        pattern = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")

        for val in series:
            if pd.isna(val) or val is None:
                extracted_values.append(np.nan)
                continue

            val_str = str(val).strip()

            # If text is long (> 30 chars or > 5 words), treat as unstructured text rather than unit token
            words = val_str.split()
            if len(val_str) > 35 or len(words) > 5:
                extracted_values.append(np.nan)
                continue

            # Check for multiplier terms (Lakhs, Crores, K, Thousands, Millions, Billions)
            multiplier = 1.0
            val_lower = val_str.lower().strip()
            
            if re.search(r"\b(crore|crores|cr)\b", val_lower):
                multiplier = 10000000.0
            elif re.search(r"\b(lakh|lakhs|lac|lacs)\b", val_lower):
                multiplier = 100000.0
            elif re.search(r"\b(thousand|thousands)\b", val_lower) or re.search(r"\d\s*k\b", val_lower):
                multiplier = 1000.0
            elif re.search(r"\b(million|millions)\b", val_lower) or re.search(r"\d\s*m\b", val_lower):
                multiplier = 1000000.0
            elif re.search(r"\b(billion|billions)\b", val_lower) or re.search(r"\d\s*b\b", val_lower):
                multiplier = 1000000000.0

            matches = pattern.findall(val_str)
            if matches:
                # Take the first matched number
                raw_num = matches[0].replace(",", "")
                try:
                    num_val = float(raw_num) * multiplier
                    extracted_values.append(num_val)
                    successful_extracts += 1
                except ValueError:
                    extracted_values.append(np.nan)
            else:
                extracted_values.append(np.nan)

        ratio = successful_extracts / len(non_nulls) if len(non_nulls) > 0 else 0.0
        return pd.Series(extracted_values, index=series.index, dtype="float64"), ratio
