"""
duplicates.py
--------------
Detect duplicate or near-duplicate samples in a dataset using cosine similarity.
Clusters duplicates via union-find and outputs representative samples.
"""

import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import os

class UnionFind:
    """Union-Find structure to group duplicate indices."""
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[py] = px


def find_duplicates(embeddings, threshold=0.90, batch_size=500):
    """
    Detect near-duplicate items using cosine similarity threshold.

    Args:
        embeddings (np.ndarray): Precomputed embeddings [N, D].
        threshold (float): Cosine similarity above which two samples are considered duplicates.
        batch_size (int): To control memory usage.

    Returns:
        clusters (list[list[int]]): List of index clusters of duplicates.
    """
    n = len(embeddings)
    uf = UnionFind(n)

    for i in range(0, n, batch_size):
        j_end = min(i + batch_size, n)
        sims = cosine_similarity(embeddings[i:j_end], embeddings)
        for bi, row in enumerate(sims):
            idx_i = i + bi
            dup_idx = np.where(row > threshold)[0]
            for j in dup_idx:
                if j != idx_i:
                    uf.union(idx_i, j)

    # group by root
    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    # filter out singletons
    return [c for c in clusters.values() if len(c) > 1]


def summarize_duplicates(clusters, image_ids=None):
    """
    Summarize duplicate clusters for visualization.

    Args:
        clusters (list[list[int]]): Output of `find_duplicates`.
        image_ids (list[str]): Optional list mapping indices to filenames or IDs.

    Returns:
        list[dict]: Cluster summaries.
    """
    summaries = []
    for group in clusters:
        rep = group[0]
        members = [image_ids[i] if image_ids else i for i in group]
        summaries.append({
            "representative": members[0],
            "duplicates": members[1:],
            "count": len(members)
        })
    return summaries


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(project_root, "..", "testing", "output")

    embeddings_path = os.path.join(base_dir, "embeddings_resnet18.npy")
    image_ids_path = os.path.join(base_dir, "image_ids.txt")

    print("Loading precomputed embeddings...")
    try:
        # Load the embeddings
        embeddings = np.load(embeddings_path)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Load image IDs if available (for better summaries)
        image_ids = None
        if os.path.exists(image_ids_path):
             with open(image_ids_path, 'r') as f:
                image_ids = [line.strip() for line in f]
             print("Image IDs loaded.")

    except FileNotFoundError:
        print(f"Error: Data files not found in {base_dir}")
        print("Falling back to random data for testing.")
        # Fallback to random data for quick testing if files are missing
        embeddings = np.random.rand(100, 128)
        image_ids = None
    
    # --- Duplicate Detection ---
    
    # Use a realistic threshold for duplicate detection
    clusters = find_duplicates(embeddings, threshold=0.98, batch_size=500)
    
    # Summarize and print results
    summary = summarize_duplicates(clusters, image_ids=image_ids)
    
    print("\n  Duplicate Analysis:")
    print(f"Found {len(summary)} duplicate clusters (total samples affected: {sum(s['count'] for s in summary)}).")
    
    # Print the first 5 clusters for review
    if summary:
        print("\nTop 5 Duplicate Clusters:")
        for s in summary[:5]:
            print(f"  - Representative: {s['representative']}, Count: {s['count']}, Duplicates: {s['duplicates']}")
