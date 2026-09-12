# Visual Debugger

A tool for catching dataset problems such as duplicates, class imbalance, outliers, low-diversity clusters, and image-quality bias **before** you spend a training run finding out the hard way. Works whether your images are fully labeled, partially labeled, or not labeled at all.

## What it does

- **Embeddings, once** — every image is converted into a numeric fingerprint (via ResNet-18) that captures what's visually in it. Everything below is computed by comparing these fingerprints, not raw pixels.
- **Labeling, for any starting point:**
  - *Labeled* — skip straight to diagnostics.
  - *Semi-labeled* — k-NN label propagation fills in the gaps using whatever you've already sorted.
  - *Unlabeled* — unsupervised k-means clustering groups visually similar images so you can review and name the groups yourself.
- **Diagnostics** — duplicate detection, class-imbalance flagging, embedding-space outliers, diversity metrics (Shannon entropy, Gini index), and PCA/t-SNE/UMAP visualization.
- **Problematic-sample review** — surfaces samples that look like they're hurting training (high influence, far from their class centroid, near-duplicates) so you can decide what to keep or drop — nothing is removed automatically.
- **Group analysis, done honestly** — if you provide real subgroup metadata, it computes Label Distribution Parity (generalized to any number of classes/groups). If you don't, it says so plainly rather than inventing fake groups, and offers an image-quality-based robustness analysis instead (explicitly *not* labeled as demographic fairness).
- **Repair suggestions** — a plain-language summary of what's wrong and what to do about it.

## Setup

Always install into a dedicated environment — never into conda `base` or your system Python. Mixing environments is the single most common cause of cryptic `ImportError` issues.

```bash
conda create -n visual-debugger python=3.11 -y && conda activate visual-debugger
pip install -r requirements.txt
```

## Running it

```bash
streamlit run dashboard/app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

### Large datasets (10k+ images)

Extraction on CPU can take a while for very large datasets. For anything beyond a few thousand images, it's faster and more reliable to extract embeddings from the terminal first, then let the dashboard pick up the result:

```bash
python3 run_extraction_standalone.py path/to/your/image/folder
```
Results save directly to `outputs/`, which the dashboard loads automatically in "Run Diagnostics" mode.


## Project structure

```
dashboard/
  app.py                  # Streamlit UI — Upload & Label, Run Diagnostics
data_utils/
  loader.py               # dataset loading, metadata extraction
  labelling.py            # k-NN label propagation for semi-labeled data
  clustering.py           # k-means pseudo-labeling for unlabeled data
  embedding_cache.py      # disk cache + background-job extraction management
  extract_worker.py       # subprocess worker (isolates crashes from the app)
  quality.py              # per-image sharpness/quality checks
embedding/
  extract.py               # ResNet-18 embedding extraction
  indexer.py                # FAISS/sklearn similarity indexing
  visualize.py               # PCA/t-SNE/UMAP plots
diagnostics/
  duplicates.py, imbalance.py, outliers.py, diversity.py, robustness.py, fix_dataset.py
influence/
  influence.py              # training-influence scoring, label distribution parity
  sensitive_attr.py         # real subgroup metadata lookup (never fabricates groups)
run_extraction_standalone.py  # terminal-based extraction for large datasets
finalize_worker_output.py     # recovers a completed background job after a crash
```
