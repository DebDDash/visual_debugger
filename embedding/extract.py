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
from concurrent.futures import ThreadPoolExecutor, TimeoutError
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
    def extract_embeddings(self, image_paths, batch_size=32, progress_callback=None,
                            image_timeout=20, checkpoint_callback=None, checkpoint_every=20):
        """
        Compute embeddings for a list of image paths.
        Loads/preprocesses each batch's images in parallel threads (PIL decode
        and file I/O both release the GIL) before the forward pass, and
        reports progress via progress_callback(done, total) if given.

        image_timeout: max seconds to wait on any single image before giving
        up on it and moving on. Without this, one unreadable/unresponsive
        file (e.g. an iCloud Drive "Optimize Mac Storage" placeholder that
        never finishes downloading, a stalled network/NAS mount, or a
        corrupt file PIL can't decode) blocks the entire batch forever with
        no error and no way to tell what's wrong — exactly what "progress
        frozen at one number" looks like. Threads that time out are
        abandoned (Python can't force-kill a thread) but no longer block
        the run; the path is reported so you can find and fix the file.

        checkpoint_callback(embeddings, ids): optional, called every
        `checkpoint_every` batches with everything extracted so far, so a
        caller can persist partial progress. Without this, a hang/crash
        loses all embeddings computed up to that point, forcing a full
        restart from image 0.
        """
        embeddings, ids = [], []
        total = len(image_paths)
        skipped = []

        def _load_one(path):
            image = Image.open(path).convert("RGB")
            return path, self.transform(image)

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
            for batch_num, i in enumerate(tqdm(range(0, total, batch_size), desc="Extracting embeddings")):
                batch_paths = image_paths[i:i + batch_size]
                futures = {pool.submit(_load_one, p): p for p in batch_paths}

                batch, current_ids = [], []
                for fut, path in futures.items():
                    try:
                        _, tensor = fut.result(timeout=image_timeout)
                        batch.append(tensor)
                        current_ids.append(path)
                    except TimeoutError:
                        print(f"[WARN] Timed out after {image_timeout}s loading {path} — skipping. "
                              f"(Common causes on Mac: an iCloud Drive placeholder that won't download, "
                              f"or a stalled network/NAS mount.)")
                        skipped.append((path, "timeout"))
                    except Exception as e:
                        print(f"[WARN] Failed to load {path}: {e}")
                        skipped.append((path, str(e)))

                if batch:
                    batch_tensor = torch.stack(batch).to(self.device)
                    feats = self.model(batch_tensor).view(len(batch), -1)
                    embeddings.append(feats.cpu().numpy())
                    ids.extend(current_ids)

                if progress_callback is not None:
                    progress_callback(min(i + batch_size, total), total)

                if checkpoint_callback is not None and (batch_num + 1) % checkpoint_every == 0 and embeddings:
                    checkpoint_callback(np.vstack(embeddings), list(ids))

        if skipped:
            print(f"[WARN] Skipped {len(skipped)} unreadable/unresponsive image(s) out of {total}. "
                  f"First few: {[p for p, _ in skipped[:5]]}")

        embeddings = np.vstack(embeddings)
        return embeddings, ids

    def save_embeddings(self, embeddings, ids, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"embeddings_{self.backbone_name}.npy"), embeddings)
        with open(os.path.join(output_dir, "image_ids.txt"), "w") as f:
            f.write("\n".join(ids))
