"""
Semi-supervised labeling utilities for image datasets.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import models, transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


def get_pretrained_model(backbone="resnet18", device="cpu"):
    """
    Load pretrained model (ResNet18 or CLIP) for feature extraction.
    Device is forced to CPU for stability on macOS.
    """
    device = device or "cpu"

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

    elif backbone.lower() == "clip":
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess

    else:
        raise ValueError("Unsupported backbone. Use 'resnet18' or 'clip'.")


@torch.no_grad()
def extract_embeddings(image_paths, model, preprocess, device="cpu", use_clip=False, batch_size=8):
    """
    Extract embeddings for a list of image paths, CPU-safe and memory conservative.
    Returns (N x D) numpy array.
    """
    device = device or "cpu"
    all_embeds = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings", leave=False):
        batch_imgs = []
        for path in image_paths[i:i + batch_size]:
            try:
                img = Image.open(path).convert("RGB")
                batch_imgs.append(preprocess(img))
            except Exception:
                continue

        if not batch_imgs:
            continue

        batch = torch.stack(batch_imgs).to(device)

        if use_clip:
            features = model.encode_image(batch)
        else:
            features = model(batch)

        features = torch.nn.functional.normalize(features, dim=1)
        all_embeds.append(features.cpu().numpy())

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
                             use_faiss=True, batch_size=8):
    """
    Full pipeline. Returns (results_df, embeddings).
    - partial_labels: list-like where unlabeled entries are None or np.nan
    """
    device = "cpu"
    model, preprocess = get_pretrained_model(backbone, device=device)
    use_clip = (backbone.lower() == "clip")

    embeddings = extract_embeddings(image_paths, model, preprocess, device=device,
                                    use_clip=use_clip, batch_size=batch_size)

    if embeddings.size == 0:
        results = pd.DataFrame({
            "image_path": image_paths,
            "original_label": partial_labels,
            "pred_label": [None] * len(image_paths),
            "final_label": [None] * len(image_paths),
            "confidence": [0.0] * len(image_paths)
        })
        return results, embeddings

    pred_labels, confidences = propagate_labels(embeddings, partial_labels, k=k, use_faiss=use_faiss)
    # pred_labels only for unlabeled entries — we need arrays aligned with input length
    # Build full-length arrays
    labels = np.array(partial_labels, dtype=object)
    unlabeled_mask = ~pd.notnull(labels)
    full_pred = labels.copy()
    full_conf = np.zeros(len(labels), dtype=float)
    full_pred[unlabeled_mask] = pred_labels
    full_conf[unlabeled_mask] = confidences

    filtered = filter_pseudo_labels(full_pred, full_conf, threshold=conf_threshold)

    results = pd.DataFrame({
        "image_path": image_paths,
        "original_label": partial_labels,
        "pred_label": full_pred.tolist(),
        "final_label": filtered,
        "confidence": full_conf.tolist()
    })

    return results, embeddings
