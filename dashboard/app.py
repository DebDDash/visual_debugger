import os
import sys
import glob
import shutil
import tempfile
import zipfile
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_utils.loader import load_dataset, bulk_extract_metadata, summarize_dataset
from data_utils import visualize as viz
from embedding import visualize as embviz
from data_utils.labelling import semi_supervised_labeling
from embedding.extract import EmbeddingExtractor
from embedding.indexer import build_faiss_index_auto, find_duplicates_faiss_fast
from embedding.visualize import reduce_embeddings, plot_embedding_scatter, plot_similarity_heatmap
from diagnostics.diversity import compute_intra_class_diversity, compute_inter_class_overlap, compute_diversity_index, plot_diversity_heatmap
from diagnostics.duplicates import find_duplicates, summarize_duplicates
from diagnostics.imbalance import summarize_imbalance
from diagnostics.outliers import summarize_outliers
from diagnostics.fix_dataset import generate_repair_suggestions, apply_repairs
from diagnostics.robustness import compute_quality_properties, bucket_by_quality, evaluate_disparity
from influence.influence import compute_influence_scores, evaluate_group_fairness, identify_problematic_samples
from influence.sensitive_attr import load_sensitive_attribute

OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

st.set_page_config(
    page_title="Visual Debugger",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1117; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
.block-container { padding-top: 2rem; max-width: 1100px; }
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card { flex: 1; background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 10px; padding: 1.1rem 1.4rem; }
.metric-card .label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem; }
.metric-card .value { font-size: 1.8rem; font-weight: 700; color: #e0e0e0; }
.metric-card .sub { font-size: 0.75rem; color: #666; margin-top: 0.2rem; }
.section-header { font-size: 1.05rem; font-weight: 600; color: #c8a8f8; border-bottom: 1px solid #2a2d3a; padding-bottom: 0.4rem; margin: 1.8rem 0 1rem; }
.step-badge { display: inline-block; background: #6c3fdb; color: #fff; font-size: 0.7rem; font-weight: 700; padding: 0.15rem 0.55rem; border-radius: 999px; margin-right: 0.5rem; letter-spacing: 0.04em; vertical-align: middle; }
.pill-ok   { background:#1a3a2a; color:#4ade80; border:1px solid #4ade80; border-radius:999px; padding:0.1rem 0.7rem; font-size:0.78rem; }
.pill-warn { background:#3a2a1a; color:#fb923c; border:1px solid #fb923c; border-radius:999px; padding:0.1rem 0.7rem; font-size:0.78rem; }
.thin-rule { border:none; border-top:1px solid #2a2d3a; margin:1.5rem 0; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔬 Visual Debugger")
    st.markdown("*Pre-training dataset auditor*")
    st.markdown("---")
    mode = st.radio("Mode", ["📂  Upload & Label", "🩺  Run Diagnostics"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("#### Quick guide")
    st.markdown("1. **Upload & Label** — upload a dataset ZIP, run semi-supervised labeling, save embeddings\n2. **Run Diagnostics** — detect duplicates, imbalance, outliers, and bias-conflicting samples")
    st.markdown("---")
    st.caption("DES646 · IIT Kanpur · 2025")

mode = mode.split("  ")[-1]

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — Upload & Label
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Upload & Label":
    st.markdown("# 📂 Upload & Label Dataset")
    st.markdown("Upload a dataset ZIP (structured folders = class labels), run semi-supervised labeling, and save embeddings for diagnostics.")

    uploaded_zip = st.file_uploader("Dataset ZIP", type=["zip"], label_visibility="collapsed")

    if uploaded_zip:
        tmp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(tmp_dir, uploaded_zip.name)
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp_dir)
        st.success(f"Extracted **{uploaded_zip.name}**")

        structured = any(
            os.path.isdir(os.path.join(tmp_dir, d))
            for d in os.listdir(tmp_dir)
            if not d.endswith(".zip")
        )

        with st.spinner("Loading dataset..."):
            try:
                records = load_dataset(tmp_dir, structured=structured)
                records = bulk_extract_metadata(records, compute_histogram=False, show_progress=False)
                summary = summarize_dataset(records)
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")
                st.stop()

        n = summary["num_images"]
        n_lab = summary["num_labeled"]
        n_unlab = summary["num_unlabeled"]
        n_cls = len(summary["class_counts"])

        st.markdown(
            '<div class="metric-row">'
            f'<div class="metric-card"><div class="label">Images</div><div class="value">{n}</div></div>'
            f'<div class="metric-card"><div class="label">Labeled</div><div class="value">{n_lab}</div><div class="sub">{n_unlab} unlabeled</div></div>'
            f'<div class="metric-card"><div class="label">Classes</div><div class="value">{n_cls}</div></div>'
            '</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Image preview</div>', unsafe_allow_html=True)
        img_paths = [r["path"] for r in records if r["path"].lower().endswith((".png",".jpg",".jpeg"))][:12]
        cols = st.columns(6)
        for i, p in enumerate(img_paths):
            try:
                cols[i % 6].image(Image.open(p), use_container_width=True)
            except Exception:
                pass

        with st.expander("📊 Dataset distributions"):
            c1, c2 = st.columns(2)
            c1.pyplot(viz.plot_class_distribution(records))
            c2.pyplot(viz.plot_brightness_histogram(records))
            c3, c4 = st.columns(2)
            c3.pyplot(viz.plot_resolution_scatter(records))
            c4.pyplot(viz.plot_corruption_pie(records))

        st.markdown('<div class="section-header"><span class="step-badge">STEP 1</span> Semi-supervised labeling</div>', unsafe_allow_html=True)

        if n_unlab == 0:
            st.info("All samples already labeled — skipping label propagation.")
            final_records = records
        else:
            st.write(f"Detected **{n_unlab}** unlabeled samples out of {n}.")
            col_a, col_b, col_c = st.columns(3)
            backbone_choice = col_a.selectbox("Backbone", ["resnet18", "clip"])
            k_neighbors     = col_b.slider("Neighbors (k)", 1, 10, 5)
            conf_threshold  = col_c.slider("Confidence threshold", 0.0, 1.0, 0.7, 0.05)

            if st.button("▶ Run label propagation"):
                with st.spinner("Extracting embeddings and propagating labels..."):
                    image_paths    = [r["path"] for r in records if os.path.isfile(r["path"])]
                    partial_labels = [r.get("label") for r in records]
                    try:
                        results_df, _ = semi_supervised_labeling(
                            image_paths, partial_labels,
                            backbone=backbone_choice, k=k_neighbors,
                            conf_threshold=conf_threshold,
                        )
                        st.success("Label propagation complete.")
                        st.dataframe(results_df.head(10), use_container_width=True)
                        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                        st.download_button("⬇ Download labeled CSV", data=csv_bytes,
                                           file_name="semi_supervised_labels.csv", mime="text/csv")
                        label_map = dict(zip(results_df["image_path"], results_df["final_label"]))
                        for r in records:
                            if r.get("label") is None:
                                r["label"] = label_map.get(r["path"])
                    except Exception as e:
                        st.error(f"Label propagation failed: {e}")
            final_records = records

        st.markdown('<div class="section-header"><span class="step-badge">STEP 2</span> Extract & save embeddings</div>', unsafe_allow_html=True)
        emb_backbone = st.selectbox("Backbone for embeddings", ["resnet18", "clip"], key="emb_bb")

        if st.button("▶ Extract embeddings"):
            img_paths_all = [
                r["path"] for r in final_records
                if os.path.isfile(r["path"]) and r["path"].lower().endswith((".jpg",".jpeg",".png",".bmp"))
            ]
            with st.spinner(f"Extracting embeddings with {emb_backbone}..."):
                try:
                    extractor  = EmbeddingExtractor(backbone=emb_backbone)
                    embeddings, ids = extractor.extract_embeddings(img_paths_all)
                    extractor.save_embeddings(embeddings, ids, output_dir=OUTPUTS_DIR)

                    id_to_label = {r["path"]: str(r.get("label", "unknown")) for r in final_records}
                    labels_arr  = np.array([id_to_label.get(i, "unknown") for i in ids])
                    np.save(os.path.join(OUTPUTS_DIR, "labels.npy"), labels_arr)
                    pd.DataFrame({"path": ids, "label": labels_arr}).to_csv(
                        os.path.join(OUTPUTS_DIR, "labels.csv"), index=False)

                    # Bug B1/B2 fix: persist to session state
                    st.session_state["embeddings"] = embeddings
                    st.session_state["labels"]     = labels_arr
                    st.session_state["ids"]        = ids
                    st.success(f"Extracted **{embeddings.shape[0]}** embeddings ({embeddings.shape[1]} dims). Saved to `{OUTPUTS_DIR}/`.")
                except Exception as e:
                    st.error(f"Embedding extraction failed: {e}")

        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — Run Diagnostics
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Run Diagnostics":
    st.markdown("# 🩺 Run Diagnostics")

    def _load_from_outputs():
        # Bug fix: don't assume resnet18 — check for whichever backbone's
        # embeddings actually exist on disk (glob for embeddings_*.npy).
        candidates = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "embeddings_*.npy")))
        lbl_path = os.path.join(OUTPUTS_DIR, "labels.npy")
        ids_path = os.path.join(OUTPUTS_DIR, "image_ids.txt")
        if not candidates:
            return None, None, None
        # If more than one backbone's embeddings exist, prefer the most
        # recently written one.
        emb_path = max(candidates, key=os.path.getmtime)
        embs = np.load(emb_path)
        lbls = np.load(lbl_path, allow_pickle=True).astype(str) if os.path.exists(lbl_path) else None
        ids  = open(ids_path).read().splitlines() if os.path.exists(ids_path) else None
        return embs, lbls, ids

    # Bug B1/B2 fix: session state first, then disk, then upload
    if "embeddings" in st.session_state:
        embeddings = st.session_state["embeddings"]
        labels     = st.session_state["labels"].astype(str)
        ids        = st.session_state["ids"]
        st.info("Using embeddings from current session.")
    else:
        embeddings, labels, ids = _load_from_outputs()
        if embeddings is not None:
            st.info(f"Loaded saved embeddings from `{OUTPUTS_DIR}/` — shape {embeddings.shape}.")
        else:
            st.warning("No saved embeddings found. Upload a dataset ZIP to extract embeddings now.")
            up = st.file_uploader("Dataset ZIP", type=["zip"])
            bb = st.selectbox("Backbone", ["resnet18", "clip"])
            if up and st.button("Extract embeddings"):
                tmp2 = tempfile.mkdtemp()
                zp   = os.path.join(tmp2, up.name)
                with open(zp, "wb") as f:
                    f.write(up.read())
                with zipfile.ZipFile(zp, "r") as z:
                    z.extractall(tmp2)
                img_paths2 = []
                for root, _, files in os.walk(tmp2):
                    for fn in files:
                        if fn.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                            img_paths2.append(os.path.join(root, fn))
                with st.spinner("Extracting..."):
                    ext2 = EmbeddingExtractor(backbone=bb)
                    embeddings, ids = ext2.extract_embeddings(img_paths2)
                    ext2.save_embeddings(embeddings, ids, output_dir=OUTPUTS_DIR)
                    labels = np.array(["unknown"] * len(ids))
                    np.save(os.path.join(OUTPUTS_DIR, "labels.npy"), labels)
                    st.session_state["embeddings"] = embeddings
                    st.session_state["labels"]     = labels
                    st.session_state["ids"]        = ids
                    shutil.rmtree(tmp2, ignore_errors=True)
                    st.rerun()
            st.stop()

    unique_cls, cls_counts = np.unique(labels, return_counts=True)
    n_cls   = len(unique_cls)
    n_total = len(labels)
    balance = "balanced" if cls_counts.std() / (cls_counts.mean() + 1e-9) < 0.15 else "imbalanced"
    pill    = "pill-ok" if balance == "balanced" else "pill-warn"

    st.markdown(
        '<div class="metric-row">'
        f'<div class="metric-card"><div class="label">Samples</div><div class="value">{n_total}</div></div>'
        f'<div class="metric-card"><div class="label">Classes</div><div class="value">{n_cls}</div></div>'
        f'<div class="metric-card"><div class="label">Embedding dim</div><div class="value">{embeddings.shape[1]}</div></div>'
        f'<div class="metric-card"><div class="label">Balance</div><div class="value" style="font-size:1.1rem;padding-top:0.35rem"><span class="{pill}">{balance}</span></div></div>'
        '</div>', unsafe_allow_html=True)

    tab_dup, tab_imb, tab_out, tab_emb, tab_inf, tab_rep = st.tabs([
        "🔁 Duplicates", "⚖️ Imbalance", "🎯 Outliers",
        "🧭 Embeddings", "🔍 Influence", "🛠 Repair",
    ])

    with tab_dup:
        st.markdown('<div class="section-header">Duplicate detection</div>', unsafe_allow_html=True)
        thresh = st.slider("Cosine similarity threshold", 0.80, 1.00, 0.95, 0.01)
        if st.button("Run duplicate detection"):
            with st.spinner("Scanning for duplicates..."):
                try:
                    clusters    = find_duplicates(embeddings, threshold=thresh)
                    summary_dup = summarize_duplicates(clusters, image_ids=ids)
                    if not summary_dup:
                        st.success("No duplicates found above threshold.")
                        st.session_state["dup_report"] = {"num_duplicate_clusters": 0, "total_affected": 0}
                    else:
                        affected = sum(s["count"] for s in summary_dup)
                        st.warning(f"Found **{len(summary_dup)}** duplicate clusters ({affected} images affected).")
                        st.dataframe(pd.DataFrame(summary_dup), use_container_width=True)
                        # Bug B4 fix: bytes not path
                        st.download_button("⬇ Download duplicate report",
                                           data=pd.DataFrame(summary_dup).to_csv(index=False).encode("utf-8"),
                                           file_name="duplicates_report.csv", mime="text/csv")
                        st.session_state["dup_report"] = {"num_duplicate_clusters": len(summary_dup), "total_affected": affected}
                except Exception as e:
                    st.error(f"Duplicate detection failed: {e}")

    with tab_imb:
        st.markdown('<div class="section-header">Class imbalance analysis</div>', unsafe_allow_html=True)
        if st.button("Run imbalance analysis"):
            with st.spinner("Analysing class distribution..."):
                try:
                    report    = summarize_imbalance(labels)
                    diversity = report["diversity"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Shannon entropy",    round(diversity["shannon_entropy"], 3))
                    c2.metric("Gini index",         round(diversity["gini_index"], 3))
                    c3.metric("Effective classes",  round(diversity["effective_classes"], 2))
                    st.bar_chart(report["class_distribution"].set_index("class")["count"])
                    if report["underrepresented_classes"]:
                        st.warning(f"Underrepresented classes: {report['underrepresented_classes']}")
                    else:
                        st.success("All classes are adequately represented.")
                    st.session_state["imbalance_report"] = {
                        "class_distribution": report["class_distribution"].set_index("class")["count"].to_dict(),
                        **diversity,
                    }
                except Exception as e:
                    st.error(f"Imbalance analysis failed: {e}")

    with tab_out:
        st.markdown('<div class="section-header">Outlier detection</div>', unsafe_allow_html=True)
        if st.button("Run outlier detection"):
            with st.spinner("Detecting outliers..."):
                try:
                    out_report = summarize_outliers(embeddings, labels)
                    emb_out    = out_report["embedding_outliers"]
                    knn_scores = out_report["knn_scores"]
                    st.markdown(f"**{len(emb_out)}** potential outliers detected.")
                    st.dataframe(emb_out, use_container_width=True)
                    st.markdown("**KNN isolation scores** (higher = more isolated)")
                    st.line_chart(knn_scores)
                    st.session_state["outlier_report"] = {
                        "num_outliers": len(emb_out),
                        "mean_knn_score": float(np.mean(knn_scores)),
                    }
                except Exception as e:
                    st.error(f"Outlier detection failed: {e}")

    with tab_emb:
        st.markdown('<div class="section-header">Embedding visualisation</div>', unsafe_allow_html=True)
        method = st.selectbox("Reduction method", ["umap", "pca", "tsne"])
        if st.button("Visualise"):
            with st.spinner(f"Reducing with {method.upper()}..."):
                try:
                    reduced = reduce_embeddings(embeddings, method=method)
                    st.plotly_chart(plot_embedding_scatter(reduced, labels=labels, ids=ids,
                                    title=f"{method.upper()} Projection"), use_container_width=True)
                    st.plotly_chart(plot_similarity_heatmap(embeddings, labels=labels), use_container_width=True)
                    st.plotly_chart(embviz.plot_embedding_variance(embeddings), use_container_width=True)
                    st.plotly_chart(embviz.plot_class_balance_radar(labels), use_container_width=True)
                except Exception as e:
                    st.error(f"Visualisation failed: {e}")

    with tab_inf:
        st.markdown('<div class="section-header">Problematic samples & group analysis</div>', unsafe_allow_html=True)
        st.caption("Flags samples that may be pushing training in the wrong direction. You choose what to keep or drop — nothing is removed automatically.")

        n_real_classes = len([c for c in np.unique(labels) if str(c).lower() != "unknown"])
        if len(embeddings) > 5000:
            st.info("Large dataset detected — near-duplicate / influence scanning can take a while. Grab a coffee ☕")

        if st.button("Run problematic-sample analysis"):
            if n_real_classes < 2:
                st.error("Cannot perform — needs more than 2 real classes to compute influence.")
            else:
                with st.spinner("Scoring influence, centroid distance, and duplicates..."):
                    try:
                        result = identify_problematic_samples(embeddings, labels)
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            cand = result["candidates"]
                            st.markdown(f"**{result['n_flagged']}** candidates flagged out of {result['n_evaluated']} evaluated.")
                            st.caption(result["note"])
                            cand_display = cand.copy()
                            cand_display["reasons"] = cand_display["reasons"].apply(lambda r: ", ".join(r))
                            st.session_state["problematic_candidates"] = cand
                            st.dataframe(cand_display, use_container_width=True)
                            st.download_button("⬇ Download candidates",
                                               data=cand_display.to_csv(index=False).encode("utf-8"),
                                               file_name="problematic_candidates.csv", mime="text/csv")
                            st.session_state["influence_report"] = {"n_flagged": result["n_flagged"]}
                    except Exception as e:
                        st.error(f"Problematic-sample analysis failed: {e}")

        if "problematic_candidates" in st.session_state:
            st.markdown("**Review candidates**")
            cand = st.session_state["problematic_candidates"]
            decisions = {}
            for _, row in cand.head(30).iterrows():
                c1, c2 = st.columns([4, 1])
                c1.write(f"Index {row['index']} · label `{row['label']}` · score {row['problem_score']:.2f} · {', '.join(row['reasons'])}")
                decisions[row["index"]] = c2.selectbox("Action", ["Keep", "Drop"], key=f"decision_{row['index']}", label_visibility="collapsed")
            if st.button("Apply keep/drop decisions"):
                drop_idx = [i for i, d in decisions.items() if d == "Drop"]
                st.session_state["dropped_indices"] = drop_idx
                st.success(f"Marked {len(drop_idx)} samples for removal. (Wire this into your training split as needed.)")

        st.markdown('<hr class="thin-rule">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Group-level analysis</div>', unsafe_allow_html=True)

        sensitive_attr = load_sensitive_attribute(expected_n=len(embeddings))

        if sensitive_attr is not None:
            fairness = evaluate_group_fairness(labels, sensitive_attr)
            st.markdown(f"**Metric: {fairness['metric_name']}**")
            st.metric("Label distribution parity (max group gap)", fairness["label_distribution_parity"])
            st.caption("0 = identical label distributions across groups; closer to 1 = groups differ sharply in which classes they contain. This is a data-composition signal, not a model-prediction fairness metric.")
        else:
            st.warning("Sensitive-group fairness unavailable — no subgroup metadata was provided. "
                       "Demographic fairness metrics cannot be reliably computed without real group membership labels.")
            st.markdown("Running an **image-quality robustness analysis** instead (not a fairness metric):")
            if st.button("Run robustness / dataset bias analysis"):
                with st.spinner("Computing per-image quality properties..."):
                    try:
                        img_paths_valid = [p for p in ids if os.path.isfile(p)] if ids else []
                        if not img_paths_valid:
                            st.error("Original image files not found on disk — can't compute quality properties from saved embeddings alone.")
                        else:
                            qdf = compute_quality_properties(img_paths_valid, show_progress=False)
                            qdf = bucket_by_quality(qdf)
                            # proxy "performance" signal: distance from own class centroid
                            # (lower = more typical/canonical for its class)
                            id_to_idx = {p: i for i, p in enumerate(ids)}
                            qdf["centroid_distance"] = qdf["path"].map(
                                lambda p: float(np.linalg.norm(embeddings[id_to_idx[p]] - embeddings[[id_to_idx[q] for q in ids if labels[id_to_idx[q]] == labels[id_to_idx[p]]]].mean(axis=0)))
                                if p in id_to_idx else np.nan
                            )
                            disparity = evaluate_disparity(qdf, "centroid_distance")
                            st.markdown(f"**{disparity['label']}**")
                            st.dataframe(pd.DataFrame(disparity["per_bucket"]).T, use_container_width=True)
                            st.metric("Max disparity across quality tiers", round(disparity["max_disparity"], 4))
                            st.caption(disparity["note"])
                            if disparity["flag"]:
                                st.warning("Notable disparity across image-quality tiers — low-quality images may behave differently in training.")
                    except Exception as e:
                        st.error(f"Robustness analysis failed: {e}")

    with tab_rep:
        st.markdown('<div class="section-header">Dataset repair suggestions</div>', unsafe_allow_html=True)
        if st.button("Generate repair suggestions"):
            metrics = {}
            if "imbalance_report" in st.session_state:
                metrics["class_distribution"] = st.session_state["imbalance_report"].get("class_distribution", {})
            if "influence_report" in st.session_state:
                metrics["high_influence_count"] = st.session_state["influence_report"].get("high_influence_count", 0)
                metrics["fairness"]             = st.session_state["influence_report"].get("fairness", {})
            if "dup_report" in st.session_state:
                metrics["duplicate_clusters"]   = st.session_state["dup_report"].get("num_duplicate_clusters", 0)
            report_path = os.path.join(OUTPUTS_DIR, "diagnostics_report.json")
            with open(report_path, "w") as f:
                json.dump({"metrics": metrics}, f, indent=2)
            suggestions = generate_repair_suggestions(report_path)
            if "error" in suggestions:
                st.error(suggestions["error"])
            else:
                for s in suggestions["suggestions"]:
                    st.markdown(f"- {s}")

        st.markdown('<hr class="thin-rule">', unsafe_allow_html=True)
        st.markdown("**Apply fixes** (class rebalancing + shuffle)")
        if st.button("Apply fixes"):
            if ids is not None and labels is not None:
                # Bug B3 fix: full DataFrame, not 15-row preview
                full_df = pd.DataFrame({"path": ids, "label": labels})
                try:
                    repaired_path = apply_repairs(full_df, label_col="label")
                    with open(repaired_path, "rb") as f:
                        st.download_button("Download repaired dataset CSV",
                                           data=f.read(),
                                           file_name="repaired_dataset.csv", mime="text/csv")
                    st.success("Repaired dataset ready.")
                except Exception as e:
                    st.error(f"Repair failed: {e}")
            else:
                st.warning("No label data found. Run diagnostics first.")
