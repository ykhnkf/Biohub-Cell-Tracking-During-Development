from __future__ import annotations

import json
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, KERNEL_ID, KERNEL_TITLE, SUPPORT_DATASET
from build_cv_audit_kernel_v4 import build_worker as build_worker_v4


def build_worker() -> str:
    worker = build_worker_v4()
    replacements = {
        'OFFICIAL / "src" / "tracking_cellmot" / "metrics.py"': 'OFFICIAL / "src" / "biohub_tracking" / "metrics.py"',
        'from tracking_cellmot.io import DEFAULT_SCALE, open_dataset': 'from biohub_tracking.io import DEFAULT_SCALE, open_dataset',
        'from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise': 'from biohub_tracking.metrics import evaluate, node_recall, per_sample_metrics, summarise',
    }
    for old, new in replacements.items():
        if old not in worker:
            raise RuntimeError(f"Expected v4 text not found: {old}")
        worker = worker.replace(old, new)
    return worker


def main() -> None:
    out = Path(".kaggle_cv_audit_kernel")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_audit.py").write_text(build_worker(), encoding="utf-8")
    metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "run_audit.py",
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
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({
        "kernel": KERNEL_ID,
        "support_dataset": SUPPORT_DATASET,
        "evaluator_package": "biohub_tracking",
        "evaluator_source": "support-pack repo/repo.zip",
    }, indent=2))


if __name__ == "__main__":
    main()
