"""
robustness.py
--------------
Image-Quality Robustness / Dataset Bias Analysis.

This is deliberately NOT called "fairness" — it says nothing about protected
demographic groups. It answers a narrower, well-supported question: "does
this dataset/model behave differently on low- vs high-quality images?" Low
quality here means measurable image properties (blur, noise, low resolution,
poor contrast, compression artifacts, saturation extremes) — not anything
about who or what is depicted.

Use this when no real sensitive-attribute metadata is available (see
influence/sensitive_attr.py::load_sensitive_attribute, which returns None
rather than fabricating groups in that case).

Pipeline:
    1. compute_quality_properties(image_paths)  -> per-image raw metrics
    2. bucket_by_quality(df)                    -> Low / Medium / High tiers
    3. evaluate_disparity(df, performance_col)  -> disparity report across tiers
"""

import os
import numpy as np
import pandas as pd
import cv2


def _safe_read_gray_bgr(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


def _estimate_noise(gray):
    """Fast noise estimate: median-filter residual standard deviation."""
    median = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.float32) - median.astype(np.float32)
    return float(np.std(residual))


def _estimate_compression_artifact(gray):
    """
    Rough JPEG blockiness proxy: mean absolute difference across 8x8 block
    boundaries vs. within blocks. Higher = more visible block artifacts.
    """
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0
    g = gray.astype(np.float32)
    # differences across vertical block boundaries (every 8th column)
    boundary_cols = np.arange(8, w - 1, 8)
    if len(boundary_cols) == 0:
        return 0.0
    boundary_diff = np.mean(np.abs(g[:, boundary_cols] - g[:, boundary_cols - 1]))
    non_boundary_cols = np.setdiff1d(np.arange(1, w), boundary_cols)
    non_boundary_diff = np.mean(np.abs(g[:, non_boundary_cols] - g[:, non_boundary_cols - 1]))
    return float(boundary_diff - non_boundary_diff)


def compute_quality_properties(image_paths, show_progress=True):
    """
    Compute brightness, contrast, blur, resolution, noise, compression
    artifact proxy, and saturation for each image.

    Returns:
        pd.DataFrame indexed by image path with one column per property.
    """
    from tqdm import tqdm
    records = []
    iterator = tqdm(image_paths, disable=not show_progress, desc="Quality metrics")
    for path in iterator:
        bgr, gray = _safe_read_gray_bgr(path)
        if bgr is None:
            records.append({"path": path, "is_corrupt": True})
            continue

        h, w = gray.shape
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        noise = _estimate_noise(gray)
        compression = _estimate_compression_artifact(gray)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        saturation = float(np.mean(hsv[:, :, 1]))

        records.append({
            "path": path,
            "is_corrupt": False,
            "resolution": w * h,
            "width": w,
            "height": h,
            "brightness": brightness,
            "contrast": contrast,
            "blur": blur,          # variance of Laplacian; LOWER = blurrier
            "noise": noise,        # higher = noisier
            "compression": compression,  # higher = more block artifacts
            "saturation": saturation,
        })
    return pd.DataFrame(records)


def _zscore(series):
    s = series.astype(float)
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def bucket_by_quality(df, n_buckets=3, labels=("Low", "Medium", "High")):
    """
    Combine the raw properties into a single composite quality score and
    bucket samples into Low/Medium/High tiers via quantiles.

    Composite score rewards: sharper (higher blur var), higher resolution,
    higher contrast, lower noise, lower compression artifacts, and
    penalizes extreme (very low/high) brightness and saturation.
    """
    df = df[~df.get("is_corrupt", False)].copy()
    if df.empty:
        return df

    score = (
        _zscore(df["blur"])
        + _zscore(df["resolution"])
        + _zscore(df["contrast"])
        - _zscore(df["noise"])
        - _zscore(df["compression"])
        - (_zscore(df["brightness"]).abs())
        - (_zscore(df["saturation"]).abs())
    )
    df["quality_score"] = score
    try:
        df["quality_bucket"] = pd.qcut(score, q=n_buckets, labels=labels, duplicates="drop")
    except ValueError:
        # too few unique values to form n_buckets quantile bins
        df["quality_bucket"] = labels[len(labels) // 2]
    return df


def evaluate_disparity(df, performance_col):
    """
    Compare a per-sample performance/behavior proxy (e.g. pseudo-label
    confidence, embedding-centroid distance, or real model accuracy if you
    have it) across quality buckets.

    This is a ROBUSTNESS / DATASET BIAS report, not a fairness report — it
    says nothing about protected demographic groups.

    Args:
        df: output of bucket_by_quality(), must contain performance_col.
        performance_col: name of the numeric column to compare across buckets.

    Returns:
        dict with per-bucket stats and a disparity summary.
    """
    if "quality_bucket" not in df.columns:
        raise ValueError("Run bucket_by_quality() first.")

    grouped = df.groupby("quality_bucket")[performance_col].agg(["mean", "std", "count"])
    max_gap = grouped["mean"].max() - grouped["mean"].min()

    return {
        "label": "Robustness / Dataset Bias Analysis (image-quality based)",
        "per_bucket": grouped.to_dict(orient="index"),
        "max_disparity": float(max_gap),
        "flag": bool(max_gap > 0.15 * (abs(grouped["mean"].mean()) + 1e-9)),
        "note": ("This measures whether behavior differs across image-quality "
                 "tiers (blur/noise/resolution/etc), not across demographic "
                 "groups. It is not a fairness metric."),
    }
