"""
extract_worker.py
------------------
Standalone script that runs embedding extraction in its own OS process.

WHY THIS EXISTS: a genuine segfault (e.g. from conflicting native OpenMP
runtimes) cannot be caught by Python's try/except — it kills the entire
process instantly, no matter how careful the surrounding code is. If
extraction runs inside the Streamlit server process itself, a crash takes
the whole app down with it (which is exactly the "Connection error" seen
before). Running it here, in a disposable child process instead, means a
crash only kills this worker: the parent (Streamlit) detects the child
exited abnormally, and can show a normal, recoverable error message instead
of dying itself.

This works identically on macOS, Linux, and Windows — it's standard
process isolation via Python's subprocess module, not an OS-specific fix.

Usage: python3 extract_worker.py <paths_file> <output_npz> [backbone]
  paths_file:  a text file, one image path per line (UTF-8)
  output_npz:  where results are written via np.savez (embeddings + ids)
  backbone:    "mobilenet_v3_small" (default) or "resnet18"
"""

import sys
import os
import platform

if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    if len(sys.argv) < 3:
        print("Usage: extract_worker.py <paths_file> <output_npz> [backbone]", file=sys.stderr)
        sys.exit(2)

    paths_file = sys.argv[1]
    output_npz = sys.argv[2]
    backbone = sys.argv[3] if len(sys.argv) > 3 else "mobilenet_v3_small"

    with open(paths_file, encoding="utf-8") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    import hashlib
    import numpy as np
    from embedding.extract import EmbeddingExtractor

    # Checkpoint filename is content-addressed (backbone + exact image list),
    # NOT derived from output_npz — output_npz embeds the caller's PID, which
    # is different on every restart, so a PID-based checkpoint path would be
    # unfindable the moment the very process that needed to resume relaunched.
    sig = hashlib.sha256((backbone + "\n" + "\n".join(image_paths)).encode("utf-8")).hexdigest()[:16]
    checkpoint_path = os.path.join(os.path.dirname(output_npz), f"_checkpoint_{backbone}_{sig}.npz")
    done_ids = set()
    prev_embeddings, prev_ids = [], []

    # Resume support: if a previous run of THIS SAME job (same output path)
    # left a checkpoint behind — because it was killed, hung on a bad image,
    # or the machine restarted — skip images already extracted instead of
    # starting over from image 0.
    if os.path.exists(checkpoint_path):
        try:
            ck = np.load(checkpoint_path, allow_pickle=True)
            prev_embeddings = [ck["embeddings"]]
            prev_ids = list(ck["ids"])
            done_ids = set(prev_ids)
            print(f"[INFO] Resuming from checkpoint: {len(done_ids)} images already extracted.")
        except Exception as e:
            print(f"[WARN] Could not read checkpoint ({e}), starting fresh.")

    remaining_paths = [p for p in image_paths if p not in done_ids]

    extractor = EmbeddingExtractor(backbone=backbone)
    progress_path = output_npz + ".progress"
    already_done = len(image_paths) - len(remaining_paths)

    def _progress(done, total):
        try:
            with open(progress_path, "w") as pf:
                pf.write(f"{done + already_done}/{len(image_paths)}")
        except OSError:
            pass  # progress reporting is best-effort, never worth failing the run over

    def _checkpoint(new_embeddings, new_ids):
        try:
            all_embeds = np.vstack(prev_embeddings + [new_embeddings]) if prev_embeddings else new_embeddings
            all_ids = prev_ids + new_ids
            tmp_path = checkpoint_path + ".tmp"
            np.savez(tmp_path, embeddings=all_embeds, ids=np.array(all_ids, dtype=object))
            os.replace(tmp_path, checkpoint_path)  # atomic on POSIX and Windows
        except OSError:
            pass  # checkpointing is best-effort, never worth failing the run over

    new_embeddings, new_ids = extractor.extract_embeddings(
        remaining_paths, progress_callback=_progress, checkpoint_callback=_checkpoint
    ) if remaining_paths else (np.zeros((0, 0)), [])

    embeddings = (np.vstack(prev_embeddings + [new_embeddings])
                  if prev_embeddings and new_embeddings.size else
                  (new_embeddings if new_embeddings.size else np.vstack(prev_embeddings)))
    ids = prev_ids + new_ids

    np.savez(output_npz, embeddings=embeddings, ids=np.array(ids, dtype=object))
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)  # job finished cleanly, checkpoint no longer needed
    print("DONE")


if __name__ == "__main__":
    main()
