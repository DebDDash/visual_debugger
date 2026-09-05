"""
Quantitative analysis of dataset diversity and class separation
based on precomputed embeddings and FAISS index.

Functions:
- compute_intra_class_diversity()
- compute_inter_class_overlap()
- compute_diversity_index()
- plot_diversity_heatmap()

Dependencies:
    numpy, pandas, scipy, sklearn, matplotlib (optional), faiss
"""

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import faiss
import os


# -----------------------------
# Intra-Class Diversity
# -----------------------------
def compute_intra_class_diversity(embeddings, labels):
    """
    Compute per-class diversity using mean pairwise embedding distance
    and variance (higher = more diverse class).

    Args:
        embeddings (np.ndarray): shape (N, D)
        labels (np.ndarray): shape (N,)
    Returns:
        pd.DataFrame: diversity metrics per class
    """
    classes = np.unique(labels)
    results = []

    for cls in classes:
        cls_emb = embeddings[labels == cls]
        if len(cls_emb) < 2:
            continue

        dist_matrix = pairwise_distances(cls_emb)
        mean_div = np.mean(dist_matrix)
        var_div = np.var(dist_matrix)

        results.append({
            "class": cls,
            "n_samples": len(cls_emb),
            "mean_distance": mean_div,
            "variance_distance": var_div
        })

    return pd.DataFrame(results)


# -----------------------------
#  Inter-Class Overlap
# -----------------------------
def frechet_distance(mu1, sigma1, mu2, sigma2):
    """Compute Fréchet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)


def compute_inter_class_overlap(embeddings, labels):
    """
    Estimate overlap between class distributions using Fréchet distance.
    Smaller = more overlap, larger = better separation.
    """
    classes = np.unique(labels)
    overlap_matrix = np.zeros((len(classes), len(classes)))

    stats = {}
    for cls in classes:
        cls_emb = embeddings[labels == cls]
        if len(cls_emb) < 2:
            continue
        stats[cls] = {
            "mu": np.mean(cls_emb, axis=0),
            "sigma": np.cov(cls_emb, rowvar=False)
        }

    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            if i >= j:
                continue
            emb1 = embeddings[labels == c1]
            emb2 = embeddings[labels == c2]
            if len(emb1) < 2 or len(emb2) < 2:
                continue
            dist = frechet_distance(stats[c1]["mu"], stats[c1]["sigma"],
                                    stats[c2]["mu"], stats[c2]["sigma"])
            overlap_matrix[i, j] = overlap_matrix[j, i] = dist

    return pd.DataFrame(overlap_matrix, index=classes, columns=classes)


# -----------------------------
#  Combined Diversity Index
# -----------------------------
def compute_diversity_index(embeddings, labels):
    intra = compute_intra_class_diversity(embeddings, labels)
    inter = compute_inter_class_overlap(embeddings, labels)

    mean_intra = intra["mean_distance"].mean()
    mean_inter = inter.values[np.triu_indices_from(inter, k=1)].mean()

    diversity_index = mean_intra / (mean_inter + 1e-8)
    return {
        "mean_intra_distance": mean_intra,
        "mean_inter_distance": mean_inter,
        "diversity_index": diversity_index
    }


# -----------------------------
#  Visualization
# -----------------------------
def plot_diversity_heatmap(overlap_df, cmap="viridis"):
    plt.figure(figsize=(6, 5))
    plt.imshow(overlap_df, cmap=cmap)
    plt.xticks(range(len(overlap_df.columns)), overlap_df.columns, rotation=45)
    plt.yticks(range(len(overlap_df.index)), overlap_df.index)
    plt.title("Inter-Class Overlap (Fréchet Distance)")
    plt.colorbar(label="Distance")
    plt.tight_layout()
    plt.show()

# For testing

if __name__ == "__main__":
    # Adjust your base directory
    base_dir = os.path.join("..", "testing", "output")

    embeddings_path = os.path.join(base_dir, "embeddings_resnet18.npy")
    labels_path = os.path.join(base_dir, "labels.npy")
    faiss_index_path = os.path.join(base_dir, "index.faiss")

    # Load precomputed data
    print("🔹 Loading precomputed embeddings and labels...")
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")

    # Load FAISS index (optional)
    if os.path.exists(faiss_index_path):
        print("Loading FAISS index...")
        index = faiss.read_index(faiss_index_path)
        print("FAISS index loaded:", index.ntotal, "vectors")
    else:
        print("No FAISS index found — skipping FAISS-based metrics.")

    # Compute metrics
    intra = compute_intra_class_diversity(embeddings, labels)
    inter = compute_inter_class_overlap(embeddings, labels)
    summary = compute_diversity_index(embeddings, labels)

    print("\n Intra-Class Diversity:\n", intra)
    print("\n Inter-Class Overlap:\n", inter)
    print("\n Summary Diversity Index:", summary)

    plot_diversity_heatmap(inter)
