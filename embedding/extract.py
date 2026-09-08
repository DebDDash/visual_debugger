"""
Extracts embeddings from images using a pretrained ResNet-18 backbone for
later use in bias analysis, duplicate detection, and labeling.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torchvision import models, transforms
from PIL import Image


def _auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EmbeddingExtractor:
    def __init__(self, backbone="resnet18", device=None):
        self.device = device or _auto_device()
        self.backbone_name = backbone.lower()
        self.model, self.transform = self._load_model(backbone)
        self.model.to(self.device).eval()

    def _load_model(self, backbone):
        """Loads the pretrained ResNet-18 model and its preprocessing pipeline."""
        if backbone.lower() == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove final classifier
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            raise ValueError(f"Unsupported backbone: {backbone!r}. Only 'resnet18' is supported.")
        return model, transform

    @torch.no_grad()
    def extract_embeddings(self, image_paths, batch_size=32, progress_callback=None):
        """
        Compute embeddings for a list of image paths.
        Loads/preprocesses each batch's images in parallel threads (PIL decode
        and file I/O both release the GIL) before the forward pass, and
        reports progress via progress_callback(done, total) if given.
        """
        embeddings, ids = [], []
        total = len(image_paths)

        def _load_one(path):
            try:
                image = Image.open(path).convert("RGB")
                return path, self.transform(image)
            except Exception as e:
                print(f"[WARN] Failed to load {path}: {e}")
                return path, None

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
            for i in tqdm(range(0, total, batch_size), desc="Extracting embeddings"):
                batch_paths = image_paths[i:i + batch_size]
                loaded = list(pool.map(_load_one, batch_paths))
                batch = [t for _, t in loaded if t is not None]
                current_ids = [p for p, t in loaded if t is not None]

                if batch:
                    batch_tensor = torch.stack(batch).to(self.device)
                    feats = self.model(batch_tensor).view(len(batch), -1)
                    embeddings.append(feats.cpu().numpy())
                    ids.extend(current_ids)

                if progress_callback is not None:
                    progress_callback(min(i + batch_size, total), total)

        embeddings = np.vstack(embeddings)
        return embeddings, ids

    def save_embeddings(self, embeddings, ids, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"embeddings_{self.backbone_name}.npy"), embeddings)
        with open(os.path.join(output_dir, "image_ids.txt"), "w") as f:
            f.write("\n".join(ids))
