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
  backbone:    currently only "resnet18" is supported
"""

import sys
import os
import platform

if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    if len(sys.argv) < 3:
        print("Usage: extract_worker.py <paths_file> <output_npz> [backbone]", file=sys.stderr)
        sys.exit(2)

    paths_file = sys.argv[1]
    output_npz = sys.argv[2]
    backbone = sys.argv[3] if len(sys.argv) > 3 else "resnet18"

    with open(paths_file, encoding="utf-8") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    import numpy as np
    import torch
    if platform.system() == "Darwin":
        torch.set_num_threads(1)

    from embedding.extract import EmbeddingExtractor

    extractor = EmbeddingExtractor(backbone=backbone)
    progress_path = output_npz + ".progress"

    def _progress(done, total):
        try:
            with open(progress_path, "w") as pf:
                pf.write(f"{done}/{total}")
        except OSError:
            pass  # progress reporting is best-effort, never worth failing the run over

    embeddings, ids = extractor.extract_embeddings(image_paths, progress_callback=_progress)
    np.savez(output_npz, embeddings=embeddings, ids=np.array(ids, dtype=object))
    print("DONE")


if __name__ == "__main__":
    main()
