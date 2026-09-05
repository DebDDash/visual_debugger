"""
outliers.py
------------
Detect potential outliers and low-quality images using embedding analysis
and image quality heuristics.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from skimage import io, color
import cv2
import os

def detect_embedding_outliers(embeddings, labels, top_k=5):
    """
    Identify outlier samples using per-class centroid distance and local density.

    Args:
        embeddings (np.ndarray): N x D array of image embeddings.
        labels (list or np.ndarray): Corresponding labels for each image.
        top_k (int): Number of most extreme outliers to flag per class.

    Returns:
        pd.DataFrame: Outlier info (index, label, distance_from_centroid).
    """
    unique_labels = np.unique(labels)
    outlier_records = []

    for cls in unique_labels:
        idx = np.where(labels == cls)[0]
        if len(idx) < 3:
            continue  # skip very small classes
        cls_embed = embeddings[idx]
        centroid = np.mean(cls_embed, axis=0)
        dists = np.linalg.norm(cls_embed - centroid, axis=1)
        top_idx = idx[np.argsort(-dists)[:top_k]]

        for i in range(len(top_idx)):
            outlier_records.append({
                "index": int(top_idx[i]),
                "label": cls,
                "distance_from_centroid": float(dists[np.argsort(-dists)[i]])
            })

    return pd.DataFrame(outlier_records)


def detect_knn_outliers(embeddings, n_neighbors=5):
    """
    Detect outliers based on k-nearest neighbor distance (density-based).

    Args:
        embeddings (np.ndarray): N x D array.
        n_neighbors (int): Number of neighbors to consider.

    Returns:
        np.ndarray: Outlier scores (higher = more isolated).
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    mean_dist = np.mean(distances[:, 1:], axis=1)
    return mean_dist


def quality_heuristics(image_path):
    """
    Evaluate simple image quality metrics (resolution, aspect ratio, contrast).

    Args:
        image_path (str): Path to image file.

    Returns:
        dict: {
            'resolution': (w, h),
            'aspect_ratio': float,
            'contrast': float,
            'low_quality_flag': bool
        }
    """
    try:
        img = io.imread(image_path)
        if img.ndim == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img

        h, w = gray.shape
        contrast = gray.std()
        aspect_ratio = w / h if h > 0 else 0

        low_res = (w < 64 or h < 64)
        odd_aspect = (aspect_ratio > 3 or aspect_ratio < 0.33)
        low_contrast = (contrast < 0.05)

        low_quality = low_res or odd_aspect or low_contrast

        return {
            "resolution": (w, h),
            "aspect_ratio": round(aspect_ratio, 2),
            "contrast": round(float(contrast), 3),
            "low_quality_flag": low_quality
        }
    except Exception:
        return {
            "resolution": (0, 0),
            "aspect_ratio": 0,
            "contrast": 0,
            "low_quality_flag": True
        }


def batch_quality_check(image_dir, file_list):
    """
    Run quality heuristics on a list of image files.

    Args:
        image_dir (str): Directory containing images.
        file_list (list): Filenames to check.

    Returns:
        pd.DataFrame: Summary with quality metrics per image.
    """
    records = []
    for f in file_list:
        full_path = os.path.join(image_dir, f)
        q = quality_heuristics(full_path)
        q["filename"] = f
        records.append(q)
    return pd.DataFrame(records)


def summarize_outliers(embeddings, labels, image_dir=None, file_list=None):
    """
    End-to-end summary combining embedding-based and heuristic outlier detection.

    Args:
        embeddings (np.ndarray): N x D embedding matrix.
        labels (list or np.ndarray): Class labels.
        image_dir (str): Path to image folder (optional).
        file_list (list): List of filenames (optional).

    Returns:
        dict: {
            'embedding_outliers': pd.DataFrame,
            'knn_scores': np.ndarray,
            'quality_summary': pd.DataFrame (if image_dir provided)
        }
    """
    emb_outliers = detect_embedding_outliers(embeddings, labels)
    knn_scores = detect_knn_outliers(embeddings)

    summary = {
        "embedding_outliers": emb_outliers,
        "knn_scores": knn_scores
    }

    if image_dir and file_list:
        summary["quality_summary"] = batch_quality_check(image_dir, file_list)

    return summary

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")
    data_dir = os.path.join(project_root, "testing", "output")
 
    IMAGE_DIRECTORY = os.path.join(project_root, "data")
    
    embeddings_path = os.path.join(data_dir, "embeddings_resnet18.npy")
    labels_path = os.path.join(data_dir, "labels.npy")
    image_ids_path = os.path.join(data_dir, "image_ids.txt") # List of filenames

    # --- Data Loading ---
    
    print("🔹 Loading embeddings and labels for outlier analysis...")
    try:
        embeddings = np.load(embeddings_path)
        labels = np.load(labels_path)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Labels shape: {labels.shape}")

        # Load file list for quality checks
        file_list = None
        if os.path.exists(image_ids_path):
             with open(image_ids_path, 'r') as f:
                file_list = [line.strip() for line in f]
             print(f"Image ID list loaded ({len(file_list)} files).")
        
    except FileNotFoundError:
        print("Error: Data files not found. Falling back to random data for testing.")
        # Fallback to random data if files are missing
        np.random.seed(42)
        embeddings = np.random.randn(100, 64)
        labels = np.random.choice(["cat", "dog", "fish"], 100)
        file_list = None
        IMAGE_DIRECTORY = None

    # --- Outlier Analysis ---
    print("\n Running Outlier Summary...")
    summary = summarize_outliers(
        embeddings, 
        labels, 
        image_dir=IMAGE_DIRECTORY, 
        file_list=file_list
    )
    
    # --- Print Results ---
    print("\n Embedding-Based Outliers (Farthest from Centroid):")
    print(summary["embedding_outliers"].head().to_markdown(index=False))
    
    print("\n KNN Outlier Score Sample (Higher = More Isolated):")
    print(summary["knn_scores"][:10])
    
    if "quality_summary" in summary:
        print("\n Low-Quality Image Heuristics Summary:")
        low_quality_count = summary["quality_summary"]["low_quality_flag"].sum()
        print(f"Total images flagged as low quality: {low_quality_count}")
        print(summary["quality_summary"][summary["quality_summary"]["low_quality_flag"]].head().to_markdown(index=False))
