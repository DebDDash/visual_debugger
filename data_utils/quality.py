import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


def assess_image_quality(data_dir, threshold=100.0, visualize=True):
    """
    Assess image quality in a dataset using Laplacian variance (sharpness metric).
    Automatically visualizes examples of blurry vs. good quality images.
    """
    print(f" Scanning images in: {data_dir}")
    records = []

    #  Compute Laplacian variance (sharpness)
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                img_path = os.path.join(root, file)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    variance = cv2.Laplacian(img, cv2.CV_64F).var()
                    quality_label = "blurry" if variance < threshold else "good"
                    records.append({
                        "path": img_path,
                        "variance": variance,
                        "quality_label": quality_label
                    })
                except Exception as e:
                    print(f"⚠️ Error reading {img_path}: {e}")

    df = pd.DataFrame(records)
    print(f" Analyzed {len(df)} images.")
    print(df['quality_label'].value_counts())

    # Visualization section
    if visualize and not df.empty:
        print("📊 Visualizing sample blurry vs. good quality images...")

        # Separate blurry and good subsets
        blurry_df = df[df['quality_label'] == "blurry"].sample(min(10, len(df[df['quality_label'] == 'blurry'])))
        good_df = df[df['quality_label'] == "good"].sample(min(10, len(df[df['quality_label'] == 'good'])))

        # Combine for visualization
        viz_df = pd.concat([blurry_df, good_df])
        
        #  Prepare images as NumPy arrays
        images = []
        for rec in tqdm(viz_df.to_dict(orient="records")):
            try:
                img = Image.open(rec['path']).convert("RGB")
                img = np.array(img) / 255.0  # normalize
                images.append((img, rec['quality_label']))
            except Exception as e:
                print(f"⚠️ Error loading image {rec['path']}: {e}")

    
        # Display image grid
        if images:
            fig, axes = plt.subplots(2, 10, figsize=(20, 4))
            fig.suptitle("Blurry (top) vs. Good (bottom) Image Samples", fontsize=16)
            for i in range(10):
                if i < len(blurry_df):
                    axes[0, i].imshow(images[i][0])
                    axes[0, i].set_title("Blurry", color='red', fontsize=8)
                if i < len(good_df):
                    axes[1, i].imshow(images[i + len(blurry_df)][0])
                    axes[1, i].set_title("Good", color='green', fontsize=8)
                axes[0, i].axis("off")
                axes[1, i].axis("off")
            plt.tight_layout()
            plt.show()

    return df

__all__ = ["assess_image_quality"]
