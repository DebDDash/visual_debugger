"""
Provides:
- load_dataset(path, structured=True)
- extract_metadata(image_record)
- summarize_dataset(records)
- save_labels(records, out_csv)
- sample_validation_set(records, val_frac=0.1, seed=42)

ImageRecord format (dict):
{
    'id': str,           # unique id (filename without path)
    'path': str,         # full path to image
    'label': Optional[str],
    'metadata': dict     # filled by extract_metadata
}

"""

from __future__ import annotations

import os
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from PIL import Image, ImageStat, UnidentifiedImageError
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _make_id(path: Path) -> str:
    return path.stem


_JUNK_DIR_NAMES = {"__MACOSX"}
_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
_UNLABELED_NAMES = {"unlabeled", "unannotated", "unknown", "untagged"}


def _is_junk_dir(p: Path) -> bool:
    """macOS zip metadata folders and any hidden ('.'-prefixed) directory are never real data."""
    return p.name in _JUNK_DIR_NAMES or p.name.startswith('.')


def _is_real_image_file(p: Path) -> bool:
    """Excludes macOS AppleDouble shadow files (e.g. '._photo.jpg') even though they share the extension."""
    return p.is_file() and p.suffix.lower() in _IMG_EXTS and not p.name.startswith('.')


def _find_leaf_class_dirs(node: Path):
    """
    Recursively finds the folders that actually contain images ('leaf' class
    folders), skipping any macOS/hidden junk directories, and skipping pure
    wrapper folders (folders that only contain further subfolders, no images
    of their own) rather than mislabeling everything under the wrapper's name.

    This handles both a flat layout (root/cats/, root/dogs/) and a wrapped
    layout (root/my_dataset/cats/, root/my_dataset/dogs/) — the latter is
    what you get from a zip whose contents sit inside one named top folder.

    Returns a list of (label_name, [image_paths]) tuples.
    """
    results = []
    for entry in sorted(node.iterdir()):
        if not entry.is_dir() or _is_junk_dir(entry):
            continue
        sub_dirs = [d for d in entry.iterdir() if d.is_dir() and not _is_junk_dir(d)]
        own_images = [f for f in entry.iterdir() if _is_real_image_file(f)]
        if sub_dirs and not own_images:
            # Pure wrapper folder (e.g. the zip's own name) — descend instead
            # of using its name as a label.
            results.extend(_find_leaf_class_dirs(entry))
        else:
            results.append((entry.name, own_images))
            if sub_dirs:
                # Mixed folder: has its own images AND further subfolders —
                # treat the subfolders as their own classes too.
                results.extend(_find_leaf_class_dirs(entry))
    return results


def load_dataset(root: str, structured: bool = True, label_csv: Optional[str] = None) -> List[Dict]:
    root = Path(root)
    records: List[Dict] = []

    if label_csv is not None:
        df = pd.read_csv(label_csv)
        for _, r in df.iterrows():
            if 'path' in df.columns:
                p = Path(r['path'])
            elif 'id' in df.columns:
                p = root / r['id']
            else:
                raise ValueError('label_csv must contain either `path` or `id` column')
            rec = {'id': _make_id(p), 'path': str(p), 'label': r.get('label', None), 'metadata': {}}
            records.append(rec)
        return records

    if structured:
        for label, image_paths in _find_leaf_class_dirs(root):
            is_unlabeled_folder = label.lower() in _UNLABELED_NAMES
            for img_path in image_paths:
                records.append({
                    'id': _make_id(img_path),
                    'path': str(img_path),
                    'label': None if is_unlabeled_folder else label,
                    'metadata': {},
                })

        # Also pick up any loose images sitting directly at root as unlabeled
        # (skip macOS AppleDouble shadow files here too).
        for img_path in root.iterdir():
            if _is_real_image_file(img_path):
                records.append({'id': _make_id(img_path), 'path': str(img_path), 'label': None, 'metadata': {}})

    else:
        # non-structured: all images unlabeled (still skip junk/hidden files)
        for img_path in sorted(root.rglob('*')):
            if _is_real_image_file(img_path) and not any(_is_junk_dir(parent) for parent in img_path.parents):
                records.append({'id': _make_id(img_path), 'path': str(img_path), 'label': None, 'metadata': {}})

    return records


def extract_metadata(record: Dict, compute_histogram: bool = False) -> Dict:
    p = Path(record['path'])
    md: Dict = {}
    try:
        with Image.open(p) as img:
            img = img.convert('RGB')
            w, h = img.size
            md['width'] = w
            md['height'] = h
            md['aspect_ratio'] = float(w) / float(h) if h != 0 else None
            stat = ImageStat.Stat(img)
            md['mean_brightness'] = float(np.mean(stat.mean))
            md['std_brightness'] = float(np.mean(stat.stddev))
            md['num_channels'] = len(img.getbands())
            md['mode'] = img.mode
            md['is_corrupt'] = False
            if compute_histogram:
                hist = img.histogram()
                hist = np.array(hist, dtype=np.float32)
                hist = hist / (hist.sum() + 1e-9)
                md['color_histogram'] = hist.tolist()
    except (UnidentifiedImageError, OSError) as e:
        md['is_corrupt'] = True
        md['error'] = str(e)
    record['metadata'] = md
    return md


def bulk_extract_metadata(records: List[Dict], compute_histogram: bool = False, show_progress: bool = True) -> List[Dict]:
    for rec in tqdm(records, disable=not show_progress):
        if not rec.get('metadata'):
            try:
                extract_metadata(rec, compute_histogram=compute_histogram)
            except Exception as e:
                rec['metadata'] = {'is_corrupt': True, 'error': str(e)}
    return records


def summarize_dataset(records: List[Dict]) -> Dict:
    n = len(records)
    labels = [r['label'] for r in records if r.get('label') is not None]
    unlabeled = [r for r in records if r.get('label') is None]
    classes = pd.Series([l for l in labels]).value_counts().to_dict() if labels else {}

    widths = [r['metadata'].get('width') for r in records if r.get('metadata') and r['metadata'].get('width')]
    heights = [r['metadata'].get('height') for r in records if r.get('metadata') and r['metadata'].get('height')]

    summary = {
        'num_images': n,
        'num_labeled': len(labels),
        'num_unlabeled': len(unlabeled),
        'class_counts': classes,
        'resolution_mean': {
            'width': float(np.mean(widths)) if widths else None,
            'height': float(np.mean(heights)) if heights else None
        },
        'resolution_median': {
            'width': float(np.median(widths)) if widths else None,
            'height': float(np.median(heights)) if heights else None
        }
    }
    return summary


def save_labels(records: List[Dict], out_csv: str) -> None:
    rows = []
    for r in records:
        rows.append({'id': r['id'], 'path': r['path'], 'label': r.get('label', None)})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)


def sample_validation_set(records: List[Dict], val_frac: float = 0.1, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    labeled = [r for r in records if r.get('label') is not None]
    unlabeled = [r for r in records if r.get('label') is None]

    if labeled:
        df = pd.DataFrame([{'id': r['id'], 'label': r['label']} for r in labeled])
        train_ids, val_ids = train_test_split(
            df['id'], test_size=val_frac, random_state=seed, stratify=df['label']
        )
        train = [r for r in records if r['id'] in set(train_ids)]
        val = [r for r in records if r['id'] in set(val_ids)]
    else:
        train, val = train_test_split(records, test_size=val_frac, random_state=seed)

    if unlabeled:
        train.extend(unlabeled)
    return train, val


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--structured', action='store_true')
    args = parser.parse_args()
    recs = load_dataset(args.dataset, structured=args.structured)
    print(f'Loaded {len(recs)} images')
    recs = bulk_extract_metadata(recs, compute_histogram=False)
    print('Summary:')
    print(json.dumps(summarize_dataset(recs), indent=2))
