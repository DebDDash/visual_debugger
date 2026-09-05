"""
sensitive_attr.py
------------------
Looks for REAL subgroup metadata (a genuine sensitive attribute column such as
gender/age/region/etc supplied by the user's own CSV). It never fabricates one.

Previously this module fell back to randomly assigning "Group A"/"Group B"
when no real column was found, and downstream fairness metrics were then
computed on those fake groups — a genuinely misleading result. That fallback
has been removed. If no real subgroup metadata exists, callers should use
`load_sensitive_attribute()` returning None and fall back to the
image-quality-based robustness analysis in diagnostics/robustness.py instead.
"""

import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.getcwd()
CSV_PATHS = [
    os.path.join(PROJECT_ROOT, "testing", "output", "labels.csv"),
    os.path.join(PROJECT_ROOT, "testing", "output", "train_labels.csv"),
    os.path.join(PROJECT_ROOT, "testing", "output", "val_labels.csv"),
    os.path.join(PROJECT_ROOT, "outputs", "labels.csv"),
]

SENSITIVE_ATTR_DIR = os.path.join(PROJECT_ROOT, "influence")
SENSITIVE_ATTR_PATH = os.path.join(SENSITIVE_ATTR_DIR, "sensitive_attr.npy")

# Common sensitive column keywords
SENSITIVE_COLUMNS = ["gender", "sex", "age", "group", "race", "region", "ethnicity", "subgroup"]


def find_real_sensitive_attribute(expected_n=None):
    """
    Look for a genuine subgroup column in any of the known label CSVs.

    Returns:
        (np.ndarray or None, str or None): (values, source_column_name).
        Returns (None, None) if no real subgroup metadata is found — this is
        an expected, valid outcome, not an error.
    """
    for csv_path in CSV_PATHS:
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        found = [c for c in df.columns if any(k in c.lower() for k in SENSITIVE_COLUMNS)]
        if found:
            col = found[0]
            values = df[col].values
            if expected_n is None or len(values) == expected_n:
                return values, col
    return None, None


def load_sensitive_attribute(expected_n=None, cache=True):
    """
    Public entry point. Tries the cached .npy first, then re-scans CSVs.
    Never fabricates a group. Returns None if unavailable.
    """
    if cache and os.path.exists(SENSITIVE_ATTR_PATH):
        cached = np.load(SENSITIVE_ATTR_PATH, allow_pickle=True)
        if expected_n is None or len(cached) == expected_n:
            return cached

    values, col = find_real_sensitive_attribute(expected_n=expected_n)
    if values is None:
        return None

    os.makedirs(SENSITIVE_ATTR_DIR, exist_ok=True)
    np.save(SENSITIVE_ATTR_PATH, values, allow_pickle=True)
    print(f"Using real sensitive attribute column '{col}' ({len(values)} samples).")
    return values


if __name__ == "__main__":
    attr = load_sensitive_attribute()
    if attr is None:
        print("No real sensitive-group metadata found. Nothing fabricated — "
              "use the robustness/quality-bucket analysis instead.")
    else:
        print(f"Loaded sensitive attribute with {len(attr)} samples, "
              f"{len(np.unique(attr))} groups: {np.unique(attr)}")
