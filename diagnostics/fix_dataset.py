# fix_dataset.py
import numpy as np
import pandas as pd
import os
import json
from sklearn.utils import resample

def generate_repair_suggestions(diagnostics_path: str):
    """
    Reads diagnostics report and returns rule-based repair suggestions.
    """
    if not os.path.exists(diagnostics_path):
        return {"error": "Diagnostics report not found"}

    with open(diagnostics_path, "r") as f:
        report = json.load(f)

    suggestions = []
    metrics = report.get("metrics", {})

    # --- Rule 1: Check class imbalance ---
    class_dist = metrics.get("class_distribution")
    if class_dist:
        max_ratio = max(class_dist.values()) / min(class_dist.values())
        if max_ratio > 1.5:
            suggestions.append("Balance class distribution using oversampling or undersampling.")

    # --- Rule 2: Check label distribution parity (only if real subgroup metadata was available) ---
    group_analysis = metrics.get("group_analysis", {})
    if group_analysis.get("available"):
        ldp = group_analysis.get("label_distribution_parity", 0)
        if ldp > 0.2:
            suggestions.append("Label distribution differs sharply across subgroups (Label Distribution Parity "
                                f"= {ldp}); consider rebalancing class representation within each group.")
    elif group_analysis.get("robustness_flag"):
        suggestions.append("Behavior differs notably across image-quality tiers (blur/noise/resolution); "
                            "consider augmenting or re-capturing low-quality samples rather than treating this as a fairness issue.")

    # --- Rule 3: Check high influence samples ---
    if metrics.get("high_influence_count", 0) > 0:
        suggestions.append("Review or remove high-influence outliers from training data.")

    if not suggestions:
        suggestions.append("No major issues detected. Dataset appears balanced and fair.")

    return {"suggestions": suggestions}


def apply_repairs(df: pd.DataFrame, label_col: str = "label"):
    """
    Applies simple data fixes: class balancing and random shuffle.
    """
    print("⚙️ Applying dataset repairs...")

    # Balance classes if imbalanced
    counts = df[label_col].value_counts()
    if counts.max() / counts.min() > 1.5:
        majority = df[df[label_col] == counts.idxmax()]
        minority = df[df[label_col] == counts.idxmin()]
        minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
        df = pd.concat([majority, minority_upsampled])
        print("✅ Balanced class distribution")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    repaired_path = os.path.join("testing", "output", "repaired_dataset.csv")
    os.makedirs(os.path.dirname(repaired_path), exist_ok=True)
    df.to_csv(repaired_path, index=False)

    print(f"✅ Repaired dataset saved at: {repaired_path}")
    return repaired_path
