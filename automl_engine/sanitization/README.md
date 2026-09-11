# 🧹 `automl_engine/sanitization` — Regex Auto-Sanitization Module

The `sanitization` module scans dirty string features to extract hidden numeric values and normalize unit multipliers into standardized base numbers.

---

## 📄 File Inventory & Class Responsibilities

### `regex_sanitizer.py`
- **`RegexSanitizer`**:
  - Regex pattern matching:
    $$\text{Pattern} = \text{r'}([-+]?\d[\d,]*\.?\d*)\text{'}$$
  - **Unit Multiplier Conversion**:
    - `"Lakh"`, `"Lakhs"`, `"Lac"` $\rightarrow \times 100,000.0$ (`"85 Lakhs"` $\rightarrow 8,500,000.0$)
    - `"Crore"`, `"Crores"`, `"Cr"` $\rightarrow \times 10,000,000.0$ (`"1.5 Crores"` $\rightarrow 15,000,000.0$)
    - `"Thousand"`, `"Thousands"`, `"K"` $\rightarrow \times 1,000.0$ (`"45K"` $\rightarrow 45,000.0$)
    - `"Million"`, `"Millions"`, `"M"` $\rightarrow \times 1,000,000.0$ (`"2.5 Million"` $\rightarrow 2,500,000.0$)
  - Automatically reclassifies text columns into numeric data types if extractable numeric tokens exceed threshold ($\ge 50\%$).

---

## 💻 Code Example

```python
import pandas as pd
from automl_engine.sanitization import RegexSanitizer

df = pd.DataFrame({
    "area": ["1,450 sq.ft.", "850 sq ft"],
    "price": ["85 Lakhs", "1.5 Crores"]
})

sanitizer = RegexSanitizer()
df_clean, report = sanitizer.sanitize_dataframe(df)

# Output:
# df_clean['area'] -> [1450.0, 850.0]
# df_clean['price'] -> [8500000.0, 15000000.0]
```
