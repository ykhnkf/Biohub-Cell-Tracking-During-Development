from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_linking_oracle_audit_kernel import KERNEL_TITLE
from build_linking_oracle_audit_kernel_v3 import build_worker as build_worker_v3

KERNEL_ID = "ykhnkf/biohub-gt-node-linking-oracle-audit"


def build_worker() -> str:
    worker = build_worker_v3()
    old = '''    geffs = sorted(TRAIN.glob("*.geff"))
    if not geffs:
        raise RuntimeError("No train GT geffs found")
'''
    new = '''    all_geffs = sorted(TRAIN.glob("*.geff"))
    if not all_geffs:
        raise RuntimeError("No train GT geffs found")

    # Fast deterministic pilot: sample up to 20 datasets from each dataset
    # family using evenly spaced indices.  The full 199-dataset audit was
    # successfully executing but was cancelled externally after 60/199, so
    # this pilot is intentionally small enough to finish quickly while still
    # covering both 44b6 and 6bba families across the sorted dataset range.
    by_prefix = {}
    for p in all_geffs:
        by_prefix.setdefault(p.stem[:4], []).append(p)
    geffs = []
    for prefix in sorted(by_prefix):
        arr = by_prefix[prefix]
        k = min(20, len(arr))
        idxs = sorted(set(np.linspace(0, len(arr) - 1, k, dtype=int).tolist()))
        geffs.extend(arr[i] for i in idxs)
    geffs = sorted(geffs)
'''
    if old not in worker:
        raise RuntimeError("Expected GEFF discovery block not found")
    worker = worker.replace(old, new, 1)
    worker = worker.replace(
        '"status":"passed","provenance":"gt_node_oracle_geometry_only_not_model_cv","n_datasets":len(geffs),',
        '"status":"passed","provenance":"gt_node_oracle_geometry_pilot_40_not_model_cv","n_datasets":len(geffs),',
        1,
    )
    worker = worker.replace(
        '"interpretation":"All GT node coordinates are supplied to the prediction graph; only temporal edges are reconstructed by one-to-one Hungarian distance matching. This isolates association difficulty and cannot be interpreted as an end-to-end score.",',
        '"interpretation":"Pilot on a deterministic evenly-spaced sample (up to 20 datasets per prefix). All GT node coordinates are supplied to the prediction graph; only temporal edges are reconstructed by one-to-one Hungarian distance matching. This isolates association difficulty and cannot be interpreted as an end-to-end score.",',
        1,
    )
    return worker


def main() -> None:
    out = Path(".kaggle_linking_oracle_kernel")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "run_linking_oracle.py").write_text(build_worker(), encoding="utf-8")
    meta = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "run_linking_oracle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [SUPPORT_DATASET],
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"kernel": KERNEL_ID, "github_run_id": os.environ.get("GITHUB_RUN_ID", "local"), "mode": "pilot40"}, indent=2))


if __name__ == "__main__":
    main()
