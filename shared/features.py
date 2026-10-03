"""Feature contract for the real UCI Adult Income model.

Lives in ``shared`` so the API (training, /predict, /schema) and the Streamlit UI
agree on exactly which columns the model consumes. Pure Python -- no sklearn or
torch imports -- so the UI can import it cheaply.

Which ``adult_income`` columns are (and are not) model inputs is a deliberate
design decision, recorded in :data:`EXCLUDED_COLS` with the reason for each.
"""
from __future__ import annotations

from typing import Dict, List

# Model inputs. Order matters: it is the column order the preprocessor sees.
NUMERIC_COLS: List[str] = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
CATEGORICAL_COLS: List[str] = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "native_country",
]
FEATURE_COLS: List[str] = NUMERIC_COLS + CATEGORICAL_COLS

# Columns stored in adult_income but deliberately NOT given to the model.
EXCLUDED_COLS: Dict[str, str] = {
    "education": "redundant with education_num (one-to-one mapping)",
    "fnlwgt": "Census sampling weight, not a characteristic of the individual",
    "sex": "protected attribute; kept in adult_income for the fairness audit only",
    "race": "protected attribute; kept in adult_income for the fairness audit only",
}

# Protected attributes used by the SQL fairness audit (never model inputs).
# Note: excluding them does not prevent disparities -- relationship
# (Husband/Wife), marital_status and occupation can act as proxies.
PROTECTED_COLS: List[str] = ["sex", "race"]

TARGET_NAME = "income"            # text column: '<=50K' / '>50K'
TARGET_LABEL = "income_label"     # 0/1 generated column in adult_income
TARGET_CLASSES = ["<=50K", ">50K"]  # index 0 / 1

# Categorical NULLs ('?' in the raw UCI files) are imputed to this explicit
# category rather than to the most frequent value.
MISSING_CATEGORY = "Unknown"
