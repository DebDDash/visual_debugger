import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

# ---------------------------------------------------------------------
# Small Classifier
# ---------------------------------------------------------------------
class SmallClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------
# Influence Computation
# ---------------------------------------------------------------------
def compute_influence_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.01,
    epochs: int = 5,
    batch_size: int = 16,
    device: str = "cpu"
) -> np.ndarray:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn

    # Ensure embeddings are numeric
    if not np.issubdtype(np.array(embeddings).dtype, np.number):
        raise ValueError("Embeddings must be numeric")

    # Encode string labels to integers if needed
    if labels.dtype.kind not in {'i', 'u'}:  # not int or unsigned int
        unique_labels, encoded_labels = np.unique(labels, return_inverse=True)
        labels = encoded_labels

    # Convert to tensors
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)

    # Define model
    model = SmallClassifier(embeddings.shape[1], len(np.unique(labels))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Data loader
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Compute influence scores
    influence_scores = []
    for i in range(len(X)):
        model.zero_grad()
        out = model(X[i].unsqueeze(0))
        loss = criterion(out, y[i].unsqueeze(0))
        loss.backward()
        grad_norm = sum((p.grad.norm() ** 2).item() for p in model.parameters())
        influence_scores.append(grad_norm)

    return np.array(influence_scores)


# ---------------------------------------------------------------------
# Label Distribution Parity (generalized multi-class, multi-group)
# ---------------------------------------------------------------------
def compute_label_distribution_parity(labels, sensitive_attr):
    """
    Generalized replacement for the old binary demographic_parity_diff /
    equalized_odds_diff. Works for any number of classes and any number of
    sensitive groups.

    For each sensitive group, compute its label distribution (proportion of
    each class within that group). The parity gap between two groups is the
    total variation distance between their distributions:
        TVD(P, Q) = 0.5 * sum(|P(c) - Q(c)|) over classes c
    We report the maximum pairwise TVD across all group pairs — this is 0
    when every group has an identical label distribution, and grows toward 1
    as groups diverge in which classes they contain.

    This measures whether classes are distributed differently across
    subgroups (a data-composition signal), NOT model prediction fairness —
    no y_pred is used, so there's no risk of the "y_pred == y_true" trivial
    result the previous implementation had.

    Returns:
        dict: {
            "label_distribution_parity": float in [0, 1],
            "per_group_distribution": {group: {class: proportion, ...}, ...},
            "worst_pair": (group_a, group_b)
        }
    """
    df = pd.DataFrame({"label": labels, "group": sensitive_attr})
    classes = sorted(df["label"].unique())
    groups = sorted(df["group"].unique())

    dist = {}
    for g in groups:
        sub = df[df["group"] == g]
        counts = sub["label"].value_counts(normalize=True)
        dist[g] = {c: float(counts.get(c, 0.0)) for c in classes}

    worst_gap, worst_pair = 0.0, (None, None)
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1:]:
            tvd = 0.5 * sum(abs(dist[g1][c] - dist[g2][c]) for c in classes)
            if tvd > worst_gap:
                worst_gap, worst_pair = tvd, (g1, g2)

    return {
        "label_distribution_parity": round(worst_gap, 4),
        "per_group_distribution": dist,
        "worst_pair": worst_pair,
    }


def evaluate_group_fairness(labels, sensitive_attr):
    """
    Entry point used by the dashboard. `sensitive_attr` should come from
    influence.sensitive_attr.load_sensitive_attribute(), which returns None
    when no real subgroup metadata exists — it never fabricates groups.

    If sensitive_attr is None, we do NOT compute a fairness number. We
    return an explicit "unavailable" result and point the caller at the
    robustness/quality-based analysis instead.
    """
    if sensitive_attr is None or len(np.unique(sensitive_attr)) < 2:
        return {
            "available": False,
            "message": ("Sensitive-group fairness unavailable — no subgroup "
                        "metadata was provided. Demographic fairness metrics "
                        "cannot be reliably computed without real group "
                        "membership labels. Consider running the image-quality "
                        "robustness analysis (diagnostics/robustness.py) instead."),
        }

    result = compute_label_distribution_parity(labels, sensitive_attr)
    result["available"] = True
    result["metric_name"] = "Label Distribution Parity"
    return result


# ---------------------------------------------------------------------
# Problematic-sample identification (replaces identify_bias_conflicting_samples)
# ---------------------------------------------------------------------
def identify_problematic_samples(
    embeddings: np.ndarray,
    labels: np.ndarray,
    top_frac: float = 0.05,
    include_duplicates: bool = True,
    duplicate_threshold: float = 0.95,
):
    """
    Surface samples that look like they're hurting training — high gradient
    influence, far from their class centroid, and/or exact/near-duplicates —
    and hand the decision to the user rather than silently flagging "bias."

    Guard: influence/self-influence is only meaningful with >= 2 real classes
    (a linear classifier and a class centroid both need >1 class to compare
    against). If fewer than 2 real classes are present, we don't compute
    anything and say so, rather than returning a misleading number.

    Args:
        embeddings: N x D array.
        labels: length-N array of class labels (None/"unknown" excluded from
            the class count).
        top_frac: fraction of samples (by combined score) to return as
            "problematic" candidates.
        include_duplicates: also flag near-duplicate clusters as a reason.
        duplicate_threshold: cosine similarity cutoff for duplicates.

    Returns:
        dict with either {"error": ...} or {"candidates": pd.DataFrame, ...}
        The DataFrame has one row per flagged sample with an "index",
        "reasons" (list of why it was flagged), and a combined "problem_score"
        — the caller (e.g. the dashboard) presents these with keep/drop
        controls; nothing is removed automatically.
    """
    labels_arr = np.array(labels, dtype=object)
    real_classes = [c for c in np.unique(labels_arr) if c is not None and str(c).lower() != "unknown"]

    if len(real_classes) < 2:
        return {"error": "Cannot perform — needs more than 2 real classes to compute influence."}

    mask = np.isin(labels_arr, real_classes)
    sub_embeddings = embeddings[mask]
    sub_labels = labels_arr[mask]
    sub_indices = np.where(mask)[0]

    # Signal 1: gradient-based self-influence
    influence_scores = compute_influence_scores(sub_embeddings, sub_labels)

    # Signal 2: per-class centroid distance (reuses diagnostics.outliers logic)
    from diagnostics.outliers import detect_embedding_outliers
    centroid_df = detect_embedding_outliers(sub_embeddings, sub_labels, top_k=len(sub_embeddings))
    centroid_dist = np.zeros(len(sub_embeddings))
    centroid_dist[centroid_df["index"].values] = centroid_df["distance_from_centroid"].values

    combined = _zscore_arr(influence_scores) + _zscore_arr(centroid_dist)

    reasons = [[] for _ in range(len(sub_embeddings))]
    for i in np.where(_zscore_arr(influence_scores) > 1.0)[0]:
        reasons[i].append("high training influence (gradient norm)")
    for i in np.where(_zscore_arr(centroid_dist) > 1.0)[0]:
        reasons[i].append("far from class centroid")

    dup_reason_idx = set()
    if include_duplicates:
        from diagnostics.duplicates import find_duplicates
        clusters = find_duplicates(sub_embeddings, threshold=duplicate_threshold)
        for cluster in clusters:
            for i in cluster[1:]:  # keep first as representative, flag the rest
                dup_reason_idx.add(i)
                reasons[i].append("near-duplicate of another sample")
                combined[i] += 1.0  # nudge duplicates up in the ranking

    n_flag = max(1, int(top_frac * len(sub_embeddings)))
    top_local_idx = np.argsort(-combined)[:n_flag]

    result_df = pd.DataFrame({
        "index": sub_indices[top_local_idx],
        "label": sub_labels[top_local_idx],
        "problem_score": combined[top_local_idx],
        "reasons": [reasons[i] if reasons[i] else ["borderline / combined score"] for i in top_local_idx],
    }).sort_values("problem_score", ascending=False).reset_index(drop=True)

    return {
        "candidates": result_df,
        "n_flagged": len(result_df),
        "n_evaluated": len(sub_embeddings),
        "note": ("These are candidates that look like they may be pushing "
                 "training in the wrong direction (high influence, far from "
                 "their class centroid, or near-duplicates). Nothing is "
                 "removed automatically — review each and choose keep/drop."),
    }


def _zscore_arr(arr):
    arr = np.asarray(arr, dtype=float)
    std = arr.std()
    if std == 0 or np.isnan(std):
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


