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


def extract_embeddings_isolated(image_paths, backbone="resnet18", cache_dir=DEFAULT_CACHE_DIR,
                                progress_callback=None, timeout=None):
    """
    Same result as get_or_extract_embeddings(), but runs the actual
    extraction in a separate OS process instead of in-process.

    A real segfault (e.g. a native OpenMP library conflict) cannot be
    caught by Python's try/except — it kills the entire process instantly.
    Running extraction here means that if it crashes, only this disposable
    child process dies; the caller (e.g. the Streamlit app) gets a normal
    RuntimeError it can display and recover from, instead of the whole app
    going down. This works the same way on macOS, Linux, and Windows.

    Returns: (embeddings: np.ndarray, ids: list[str], was_cached: bool)
    Raises: RuntimeError if the worker process crashed or failed.
    """
    cached = load_cached_embeddings(image_paths, backbone, cache_dir)
    if cached is not None:
        return cached, list(image_paths), True

    import subprocess
    import sys
    import tempfile
    import time

    os.makedirs(cache_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(image_paths))
        paths_file = f.name

    output_npz = os.path.join(cache_dir, f"_worker_output_{os.getpid()}.npz")
    progress_path = output_npz + ".progress"
    worker_script = os.path.join(os.path.dirname(__file__), "extract_worker.py")

    proc = subprocess.Popen(
        [sys.executable, worker_script, paths_file, output_npz, backbone],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    start = time.time()
    try:
        while proc.poll() is None:
            if progress_callback is not None and os.path.exists(progress_path):
                try:
                    with open(progress_path) as pf:
                        done, total = map(int, pf.read().strip().split("/"))
                    progress_callback(done, total)
                except (ValueError, OSError):
                    pass
            if timeout is not None and (time.time() - start) > timeout:
                proc.kill()
                raise RuntimeError(f"Embedding extraction timed out after {timeout}s.")
            time.sleep(0.4)

        _, stderr = proc.communicate()
    finally:
        try:
            os.remove(paths_file)
        except OSError:
            pass
        try:
            os.remove(progress_path)
        except OSError:
            pass

    if proc.returncode != 0 or not os.path.exists(output_npz):
        # A negative return code on Unix (e.g. -11) means the process was
        # killed by a signal — SIGSEGV for a segfault specifically. On
        # Windows this shows up as a large nonzero exit code instead.
        crash_note = ""
        if proc.returncode is not None and proc.returncode < 0:
            crash_note = " (killed by a signal — this is a native crash/segfault, not a Python error in your data or code)"
        raise RuntimeError(
            f"Embedding extraction failed in the background worker "
            f"(exit code {proc.returncode}){crash_note}.\n{stderr[-2000:] if stderr else ''}"
        )

    data = np.load(output_npz, allow_pickle=True)
    embeddings = data["embeddings"]
    ids = list(data["ids"])
    try:
        os.remove(output_npz)
    except OSError:
        pass

    save_cached_embeddings(image_paths, backbone, embeddings, cache_dir)
    return embeddings, ids, False
