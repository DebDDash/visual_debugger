"""
embedding_cache.py
-------------------
Persistent, content-addressed cache for extracted image embeddings.

Extraction (running every image through a CNN forward pass) is the
expensive step in semi-supervised labeling — k-NN propagation over the
resulting vectors is cheap by comparison. Without a cache, re-tuning `k` or
the confidence threshold, or simply restarting the app, forces a full
re-extraction every time. This module makes extraction a one-time cost per
(dataset, backbone) pair.

Cache key = sha256 of (sorted image paths + each file's mtime+size + backbone
name). Using mtime+size (not just the path) means editing/replacing an image
correctly invalidates the cache instead of silently returning stale
embeddings for changed content.
"""

import os
import hashlib
import numpy as np

DEFAULT_CACHE_DIR = os.path.join(os.getcwd(), ".embedding_cache")


def _signature(image_paths, backbone):
    h = hashlib.sha256()
    h.update(backbone.encode("utf-8"))
    for p in sorted(image_paths):
        try:
            stat = os.stat(p)
            h.update(f"{p}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8"))
        except OSError:
            h.update(f"{p}:missing".encode("utf-8"))
    return h.hexdigest()


def _cache_path(image_paths, backbone, cache_dir):
    sig = _signature(image_paths, backbone)
    return os.path.join(cache_dir, f"embeddings_{backbone}_{sig[:16]}.npz")


def load_cached_embeddings(image_paths, backbone, cache_dir=DEFAULT_CACHE_DIR):
    """
    Returns an (N x D) np.ndarray aligned to `image_paths` order, or None if
    no valid cache exists for this exact dataset+backbone combination.
    """
    path = _cache_path(image_paths, backbone, cache_dir)
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        cached_paths = list(data["image_paths"])
        if cached_paths != list(image_paths):
            return None  # order or membership changed, don't risk misalignment
        return data["embeddings"]
    except Exception:
        return None


def save_cached_embeddings(image_paths, backbone, embeddings, cache_dir=DEFAULT_CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(image_paths, backbone, cache_dir)
    np.savez(path, embeddings=embeddings, image_paths=np.array(image_paths, dtype=object))
    return path


def get_or_extract_embeddings(image_paths, backbone, extract_fn, cache_dir=DEFAULT_CACHE_DIR):
    """
    extract_fn: callable(image_paths) -> np.ndarray, called only on a cache miss.
    """
    cached = load_cached_embeddings(image_paths, backbone, cache_dir)
    if cached is not None:
        return cached, True  # (embeddings, was_cached)
    embeddings = extract_fn(image_paths)
    save_cached_embeddings(image_paths, backbone, embeddings, cache_dir)
    return embeddings, False
