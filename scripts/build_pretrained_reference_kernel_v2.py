from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_pretrained_reference_kernel import (
    COMPETITION, KERNEL_ID, KERNEL_TITLE, SUPPORT_DATASET,
    build_worker as build_worker_v1,
)


def build_worker() -> str:
    worker = build_worker_v1()
    old = '''            "edge_jaccard": float(pm["edge_jaccard"]),
            "adjusted_edge_jaccard": float(pm["adjusted_edge_jaccard"]),
            "division_jaccard": float(pm["division_jaccard"]),
            "score": float(pm["score"]),
'''
    new = '''            "edge_jaccard": float(pm["edge_jaccard"]),
            "adjusted_edge_jaccard": float(pm["adj_edge_jaccard"]),
            "division_jaccard": (
                float(er.division_tp) / float(er.division_tp + er.division_fp + er.division_fn)
                if (er.division_tp + er.division_fp + er.division_fn) > 0 else float("nan")
            ),
            "score": (
                float(pm["adj_edge_jaccard"]) + 0.1 * float(er.division_tp) / float(er.division_tp + er.division_fp + er.division_fn)
                if (er.division_tp + er.division_fp + er.division_fn) > 0 else float(pm["adj_edge_jaccard"])
            ),
'''
    if old not in worker:
        raise RuntimeError("Expected v1 per-sample metric block not found")
    return worker.replace(old, new, 1)


def main() -> None:
    out = Path(".kaggle_pretrained_reference_kernel")
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "run_reference.py").write_text(build_worker(), encoding="utf-8")
    meta = {
        "id": KERNEL_ID, "title": KERNEL_TITLE, "code_file": "run_reference.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": [], "dataset_sources": [SUPPORT_DATASET], "kernel_sources": [],
        "competition_sources": [COMPETITION], "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"kernel": KERNEL_ID, "github_run_id": os.environ.get("GITHUB_RUN_ID", "local")}, indent=2))

if __name__ == "__main__": main()
