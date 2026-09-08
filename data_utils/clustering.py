"""
clustering.py
--------------
Unsupervised pseudo-labeling for datasets with ZERO real labels.

k-NN label propagation (data_utils/labelling.py) needs at least a handful of
labeled examples per class to propagate from — with zero labels there's
nothing to propagate from. This module instead clusters the embedding space
directly: the user supplies how many classes they expect, we run k-means,
and each sample gets assigned to the cluster nearest it.

IMPORTANT framing: these are CLUSTER assignments, not verified ground-truth
labels. Clusters reflect visual similarity in the pretrained backbone's
embedding space, which usually correlates with real object categories but
isn't guaranteed to align with what the user actually means by "class" —
e.g. clusters might split on lighting/background instead of the object.
Always present these as "cluster_0", "cluster_1", ... for the user to
inspect, rename, or merge — never silently relabel them as if verified.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int, random_state: int = 42):
    """
    Cluster embeddings into n_clusters groups via k-means.

    Args:
        embeddings: N x D array.
        n_clusters: how many classes the user expects. Must be >= 2 and
            < N (k-means needs more samples than clusters).

    Returns:
        dict: {
            "cluster_ids": np.ndarray of shape (N,), values 0..n_clusters-1
            "silhouette": float in [-1, 1], higher = better-separated
                clusters. Rough guide: >0.5 good separation, <0.2 weak/
                overlapping clusters (the number of classes you picked may
                not match the data's real structure).
            "cluster_sizes": {cluster_id: count}
        }
    """
    n = len(embeddings)
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    if n_clusters >= n:
        raise ValueError(f"n_clusters ({n_clusters}) must be less than the number of samples ({n}).")

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = km.fit_predict(embeddings)

    try:
        sil = float(silhouette_score(embeddings, cluster_ids))
    except Exception:
        sil = None  # can fail with degenerate clusters (e.g. a cluster of size 1)

    sizes = {int(c): int((cluster_ids == c).sum()) for c in np.unique(cluster_ids)}

    return {
        "cluster_ids": cluster_ids,
        "silhouette": sil,
        "cluster_sizes": sizes,
    }


def build_pseudo_label_df(image_paths, cluster_ids, cluster_names=None):
    """
    Build the results dataframe, with human-editable cluster names.

    Args:
        cluster_names: optional dict {cluster_id: "your chosen name"}.
            Falls back to "cluster_0", "cluster_1", ... for any cluster
            without an explicit name.
    """
    cluster_names = cluster_names or {}
    labels = [cluster_names.get(int(c), f"cluster_{c}") for c in cluster_ids]
    return pd.DataFrame({
        "image_path": image_paths,
        "cluster_id": cluster_ids,
        "pseudo_label": labels,
    })
