"""
Extracts embeddings from images using pretrained backbones (ResNet18 or CLIP) for later use in bias analysis, duplicate detection, and labeling.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import models, transforms
from PIL import Image
 

class EmbeddingExtractor:
    def __init__(self, backbone="resnet18", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone_name = backbone.lower()
        self.model, self.transform = self._load_model(backbone)
        self.model.to(self.device).eval()

    def _load_model(self, backbone):
        """Loads the desired pretrained model and preprocessing pipeline."""
        if backbone == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove final classifier
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif backbone == "clip":
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            transform = preprocess
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        return model, transform

    @torch.no_grad()
    def extract_embeddings(self, image_paths, batch_size=32):
        """Compute embeddings for a list of image paths."""
        embeddings, ids = [], []
        batch, current_ids = [], []

        for img_path in tqdm(image_paths, desc="Extracting embeddings"):
            try:
                image = Image.open(img_path).convert("RGB")
                tensor = self.transform(image)
                batch.append(tensor)
                current_ids.append(img_path)
            except Exception as e:
                print(f"[WARN] Failed to load {img_path}: {e}")
                continue

            if len(batch) == batch_size:
                batch_tensor = torch.stack(batch).to(self.device)
                feats = self.model(batch_tensor).view(len(batch), -1)
                embeddings.append(feats.cpu().numpy())
                ids.extend(current_ids)
                batch, current_ids = [], []

        if batch:
            batch_tensor = torch.stack(batch).to(self.device)
            feats = self.model(batch_tensor).view(len(batch), -1)
            embeddings.append(feats.cpu().numpy())
            ids.extend(current_ids)

        embeddings = np.vstack(embeddings)
        return embeddings, ids

    def save_embeddings(self, embeddings, ids, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"embeddings_{self.backbone_name}.npy"), embeddings)
        with open(os.path.join(output_dir, "image_ids.txt"), "w") as f:
            f.write("\n".join(ids))
