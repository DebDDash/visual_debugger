"""
Semi-supervised labeling utilities for image datasets.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torchvision import models, transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


def get_pretrained_model(backbone="resnet18", device=None):
    """
    Load pretrained ResNet-18 model for feature extraction.
    Auto-detects CUDA/MPS if available, falling back to CPU only when
    neither exists.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if backbone.lower() == "resnet18":
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Identity()
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return model.to(device).eval(), preprocess

    else:
        raise ValueError(f"Unsupported backbone: {backbone!r}. Only 'resnet18' is supported.")


@torch.no_grad()
def extract_embeddings(image_paths, model, preprocess, device="cpu", batch_size=32, progress_callback=None):
    """
    Extract embeddings for a list of image paths. Returns (N x D) numpy array.

    Loads and preprocesses each batch's images in parallel threads before
    running the model forward pass — PIL's JPEG decoding and file I/O both
    release Python's GIL, so this helps meaningfully even with no GPU at all,
    on top of whatever the model forward pass itself gets from `device`.

    progress_callback: optional callable(done, total) invoked after each
    batch, so a caller (e.g. a Streamlit progress bar) can show real
    progress instead of a silent spinner with no sense of how far along or
    how much longer a large dataset will take.
    """
    device = device or "cpu"
    all_embeds = []
    total = len(image_paths)

    def _load_one(path):
        try:
            img = Image.open(path).convert("RGB")
            return preprocess(img)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        for i in tqdm(range(0, total, batch_size), desc="Extracting embeddings", leave=False):
            batch_paths = image_paths[i:i + batch_size]
            loaded = list(pool.map(_load_one, batch_paths))
            batch_imgs = [t for t in loaded if t is not None]

            if batch_imgs:
                batch = torch.stack(batch_imgs).to(device)
                features = model(batch)
                features = torch.nn.functional.normalize(features, dim=1)
                all_embeds.append(features.cpu().numpy())

            if progress_callback is not None:
                progress_callback(min(i + batch_size, total), total)

    if not all_embeds:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(all_embeds)


def propagate_labels(embeddings, labels, k=5, use_faiss=True):
    """
    Propagate labels from labeled to unlabeled samples using NN.
    Returns lists: predicted_labels, confidences (aligned to unlabeled indices).
    """
    labels = np.array(labels, dtype=object)
    if embeddings.size == 0:
        return [], []

    labeled_mask = pd.notnull(labels)
    unlabeled_mask = ~labeled_mask

    if labeled_mask.sum() == 0:
        # nothing to propagate from
        return [None] * unlabeled_mask.sum(), [0.0] * unlabeled_mask.sum()

    X_labeled = embeddings[labeled_mask]
    y_labeled = labels[labeled_mask].astype(object)
    X_unlabeled = embeddings[unlabeled_mask]

    # FAISS fallback to sklearn if not available or on error
    if use_faiss and _HAS_FAISS:
        try:
            index = faiss.IndexFlatIP(X_labeled.shape[1])
            index.add(X_labeled.astype("float32"))
            sims, idxs = index.search(X_unlabeled.astype("float32"), k)
        except Exception:
            use_faiss = False

    if not use_faiss:
        nn = NearestNeighbors(n_neighbors=min(k, len(X_labeled)), metric="cosine")
        nn.fit(X_labeled)
        idxs = nn.kneighbors(X_unlabeled, n_neighbors=min(k, len(X_labeled)), return_distance=False)

    pred_labels, confidences = [], []
    for neighbor_idxs in idxs:
        neighbor_labels = y_labeled[neighbor_idxs]
        valid_labels = [l for l in neighbor_labels if pd.notnull(l)]

        if not valid_labels:
            pred_labels.append(None)
            confidences.append(0.0)
            continue

        unique, counts = np.unique(valid_labels, return_counts=True)
        best_label = unique[np.argmax(counts)]
        confidence = counts.max() / counts.sum()
        pred_labels.append(best_label)
        confidences.append(float(confidence))

    return pred_labels, confidences


def filter_pseudo_labels(pred_labels, confidences, threshold=0.7):
    final_labels = [lbl if (conf is not None and conf >= threshold) else None
                    for lbl, conf in zip(pred_labels, confidences)]
    return final_labels


def semi_supervised_labeling(image_paths, partial_labels, backbone="resnet18", k=5, conf_threshold=0.7,
                             use_faiss=True, batch_size=32, progress_callback=None,
                             use_disk_cache=True, cache_dir=None):
    """
    Full pipeline. Returns (results_df, embeddings, device_used).
    - partial_labels: list-like where unlabeled entries are None or np.nan
    - progress_callback: optional callable(done, total), forwarded to extract_embeddings
    - use_disk_cache: if True, reuse a persistent on-disk embedding cache keyed by
      (dataset content, backbone) so re-running with different k/conf_threshold,
      or restarting the app entirely, never re-extracts embeddings unnecessarily.
    """
    from data_utils.embedding_cache import get_or_extract_embeddings, DEFAULT_CACHE_DIR
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    model, preprocess = get_pretrained_model(backbone, device=None)  # auto-detect CUDA/MPS
    device = next(model.parameters()).device.type

    def _extract(paths):
        return extract_embeddings(paths, model, preprocess, device=device,
                                  batch_size=batch_size,
                                  progress_callback=progress_callback)

    if use_disk_cache:
        embeddings, was_cached = get_or_extract_embeddings(image_paths, backbone, _extract, cache_dir=cache_dir)
        if was_cached and progress_callback is not None:
            progress_callback(len(image_paths), len(image_paths))  # jump progress bar to done
    else:
        embeddings = _extract(image_paths)

    if embeddings.size == 0:
        results = pd.DataFrame({
            "image_path": image_paths,
            "original_label": partial_labels,
            "pred_label": [None] * len(image_paths),
            "final_label": [None] * len(image_paths),
            "confidence": [0.0] * len(image_paths)
        })
        return results, embeddings, device

    results = relabel_from_embeddings(image_paths, partial_labels, embeddings, k=k, conf_threshold=conf_threshold, use_faiss=use_faiss)
    return results, embeddings, device


def relabel_from_embeddings(image_paths, partial_labels, embeddings, k=5, conf_threshold=0.7, use_faiss=True):
    """
    Re-run just the propagation + filtering step against ALREADY-EXTRACTED
    embeddings. Use this when a user is only retuning k or the confidence
    threshold — it's near-instant since it skips the CNN forward pass
    entirely. Call this instead of semi_supervised_labeling() for that case.
    """
    pred_labels, confidences = propagate_labels(embeddings, partial_labels, k=k, use_faiss=use_faiss)

    labels = np.array(partial_labels, dtype=object)
    unlabeled_mask = ~pd.notnull(labels)
    full_pred = labels.copy()
    full_conf = np.zeros(len(labels), dtype=float)
    full_pred[unlabeled_mask] = pred_labels
    full_conf[unlabeled_mask] = confidences

    filtered = filter_pseudo_labels(full_pred, full_conf, threshold=conf_threshold)

    return pd.DataFrame({
        "image_path": image_paths,
        "original_label": partial_labels,
        "pred_label": full_pred.tolist(),
        "final_label": filtered,
        "confidence": full_conf.tolist()
    })
