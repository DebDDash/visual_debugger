"""
imbalance.py
-------------
Analyze class distribution, detect imbalance, and compute diversity metrics.
Generates histograms and summary statistics for the dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import entropy
import os

def compute_class_distribution(labels):
    """
    Compute raw and normalized counts per class.

    Args:
        labels (list or np.ndarray): Class labels for samples.

    Returns:
        pd.DataFrame: DataFrame with class, count, and proportion.
    """
    counts = Counter(labels)
    total = sum(counts.values())
    data = {
        "class": list(counts.keys()),
        "count": list(counts.values()),
        "proportion": [c / total for c in counts.values()]
    }
    df = pd.DataFrame(data).sort_values("count", ascending=False)
    return df


def plot_class_histogram(df, title="Class Distribution"):
    """
    Plot histogram of class counts.

    Args:
        df (pd.DataFrame): Output of compute_class_distribution().
        title (str): Plot title.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(df["class"], df["count"])
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def compute_diversity_metrics(labels):
    """
    Compute diversity metrics: Shannon entropy, Gini index, effective number of classes.

    Args:
        labels (list or np.ndarray): Class labels.

    Returns:
        dict: {
            'shannon_entropy': float,
            'gini_index': float,
            'effective_classes': float
        }
    """
    counts = np.array(list(Counter(labels).values()), dtype=np.float32)
    probs = counts / np.sum(counts)
    shannon = entropy(probs, base=2)
    gini = 1.0 - np.sum(probs ** 2)
    effective_classes = 2 ** shannon
    return {
        "shannon_entropy": round(float(shannon), 4),
        "gini_index": round(float(gini), 4),
        "effective_classes": round(float(effective_classes), 2)
    }


def flag_imbalance(df, mean_ratio=0.5):
    """
    Identify severely underrepresented classes relative to the mean class size.

    A class is flagged if its count falls below `mean_ratio` (default 50%) of
    the mean class size, where mean class size = total rows / number of classes.
    This adapts automatically to however many classes are present, unlike a
    fixed proportion cutoff which unfairly penalizes datasets with many classes
    (each class's "fair share" shrinks as n_classes grows).

    Args:
        df (pd.DataFrame): Output of compute_class_distribution().
        mean_ratio (float): Fraction of mean class size below which a class
            is considered underrepresented (default 0.5 = 50%).

    Returns:
        list[str]: Underrepresented class names.
    """
    n_classes = len(df)
    if n_classes == 0:
        return []
    mean_class_size = df["count"].sum() / n_classes
    underrepresented = df[df["count"] < mean_ratio * mean_class_size]["class"].tolist()
    return underrepresented


def summarize_imbalance(labels):
    """
    Full diagnostic summary: class stats, imbalance flags, diversity metrics.

    Args:
        labels (list or np.ndarray): Class labels.

    Returns:
        dict: summary report
    """
    df = compute_class_distribution(labels)
    diversity = compute_diversity_metrics(labels)
    under = flag_imbalance(df)
    summary = {
        "n_classes": len(df),
        "class_distribution": df,
        "underrepresented_classes": under,
        "diversity": diversity
    }
    return summary


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "testing", "output")
    labels_path = os.path.join(base_dir, "labels.npy")

    print("Loading labels for imbalance analysis...")
    try:
        # Load the labels
        labels = np.load(labels_path)
        print(f"Labels shape: {labels.shape}")

    except FileNotFoundError:
        print(f"Error: Labels file not found in {base_dir}")
        print("Falling back to random data for testing.")
        # Fallback to random data if files are missing
        labels = np.random.choice(["cat", "dog", "bird", "fish"], size=200, p=[0.5, 0.3, 0.15, 0.05])
    
    # --- Imbalance Analysis ---
    report = summarize_imbalance(labels)
    
    print("\n Imbalance and Diversity Metrics:")
    for key, value in report['diversity'].items():
        print(f"  - {key.replace('_', ' ').title()}: {value}")
        
    print(f"\n Underrepresented Classes (< 10%): {report['underrepresented_classes']}")
    
    # Plotting the result
    plot_class_histogram(report["class_distribution"], title="Actual Dataset Class Distribution")
