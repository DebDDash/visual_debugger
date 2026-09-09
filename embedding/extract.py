"""
Extracts embeddings from images using a pretrained CNN backbone (default:
MobileNetV3-Small; ResNet-18 also available) for
later use in bias analysis, duplicate detection, and labeling.
"""

import os
import random
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torchvision import models, transforms
from PIL import Image

# Fixed seed used everywhere embeddings are extracted, so re-running on the
# same images always yields the same vectors.
SEED = 42

# Backbones available for extraction. mobilenet_v3_small is the default:
# it's ~4-6x fewer FLOPs than resnet18 with only a modest quality drop,
# which barely matters for k-NN label propagation.
_BACKBONES = {
    "resnet18": {
        "weights": lambda: models.ResNet18_Weights.IMAGENET1K_V1,
        "builder": models.resnet18,
    },
    "mobilenet_v3_small": {
        "weights": lambda: models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        "builder": models.mobilenet_v3_small,
    },
}


def set_deterministic(seed=SEED):
    """
    Pins every source of randomness that can affect extracted embeddings.
    Called once per EmbeddingExtractor so results are reproducible run to
    run, machine to machine.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _resolve_device(device, backbone_name):
    """
    Extraction is disk-cached (see data_utils/embedding_cache.py), so it
    only ever runs once per (dataset, backbone) — there's no recurring cost
    to spending that one run on CPU. We default to CPU because Apple's MPS
    backend does NOT guarantee bit-identical outputs across runs (its
    conv/matmul kernels use a non-deterministic reduction order), which is
    exactly what was causing the same image to get different embeddings,
    and occasionally a different propagated label, on Mac.

    CUDA is left as an explicit opt-in (pass device="cuda") for anyone who
    wants GPU speed and is fine trading away bit-exact reproducibility;
    even then we still enable cuDNN deterministic mode below to get as
    close as PyTorch allows.
    """
    if device is not None:
        return device
    return "cpu"


class EmbeddingExtractor:
    def __init__(self, backbone="mobilenet_v3_small", device=None):
        set_deterministic()
        self.device = _resolve_device(device, backbone)
        if self.device == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.backbone_name = backbone.lower()
        self.model, self.transform = self._load_model(backbone)
        self.model.to(self.device).eval()

    def _load_model(self, backbone):
        """Loads the pretrained backbone and its preprocessing pipeline."""
        key = backbone.lower()
        if key not in _BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone!r}. "
                f"Choose one of {sorted(_BACKBONES)}."
            )
        spec = _BACKBONES[key]
        model = spec["builder"](weights=spec["weights"]())
        model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove final classifier
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
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
