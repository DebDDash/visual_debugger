# data_utils/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_class_distribution(records):
    """Bar chart of sample counts per label."""
    labels = [r.get("label") for r in records if r.get("label") is not None]
    df = pd.DataFrame(labels, columns=["label"])
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x="label", order=df['label'].value_counts().index)
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def plot_brightness_histogram(records):
    """Histogram of mean brightness across images."""
    brightness = [r["metadata"].get("mean_brightness") for r in records if r["metadata"].get("mean_brightness") is not None]
    plt.figure(figsize=(6,4))
    plt.hist(brightness, bins=30, color="skyblue", edgecolor="black")
    plt.title("Image Brightness Distribution")
    plt.xlabel("Mean Brightness")
    plt.ylabel("Frequency")
    plt.tight_layout()
    return plt.gcf()

def plot_resolution_scatter(records):
    """Scatter plot of image width vs height."""
    widths = [r["metadata"].get("width") for r in records if r["metadata"].get("width")]
    heights = [r["metadata"].get("height") for r in records if r["metadata"].get("height")]
    plt.figure(figsize=(5,5))
    plt.scatter(widths, heights, alpha=0.5)
    plt.title("Image Width vs Height")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.tight_layout()
    return plt.gcf()

def plot_corruption_pie(records):
    """Pie chart of corrupt vs valid images."""
    corrupt = sum(1 for r in records if r["metadata"].get("is_corrupt"))
    valid = len(records) - corrupt
    plt.figure(figsize=(4,4))
    plt.pie([valid, corrupt], labels=["Valid", "Corrupt"], autopct="%1.1f%%", colors=["#77dd77", "#ff6961"])
    plt.title("Corrupt vs Valid Images")
    return plt.gcf()

def plot_label_brightness_violin(records):
    """Check if some labels are systematically brighter/darker."""
    data = []
    for r in records:
        if r.get("label") and r["metadata"].get("mean_brightness"):
            data.append((r["label"], r["metadata"]["mean_brightness"]))
    df = pd.DataFrame(data, columns=["label", "brightness"])
    plt.figure(figsize=(8,4))
    sns.violinplot(data=df, x="label", y="brightness")
    plt.title("Brightness per Label")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()
