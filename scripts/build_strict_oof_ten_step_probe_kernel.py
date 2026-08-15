from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_strict_oof_one_step_probe_kernel import (
    PROBE_TRAIN,
    PROBE_VAL,
    build_worker as build_one_step_worker,
)

KERNEL_ID = "ykhnkf/biohub-strict-oof-ten-step-probe"
KERNEL_TITLE = "Biohub Strict OOF Ten Step Probe"


def build_worker() -> str:
    worker = build_one_step_worker()
    replacements = [
        ("biohub_strict_oof_one_step_probe", "biohub_strict_oof_ten_step_probe"),
        ("strict_oof_one_step_probe", "strict_oof_ten_step_probe"),
        ("one_step_splits.json", "ten_step_splits.json"),
        ("one_step_probe_summary.json", "ten_step_probe_summary.json"),
    ]
    for old, new in replacements:
        worker = worker.replace(old, new)

    # Replace both Python keyword syntax and JSON/dict syntax regardless of spacing.
    worker = re.sub(r"max_iters\s*=\s*1\b", "max_iters=10", worker)
    worker = re.sub(r'("max_iters"\s*:\s*)1\b', r'\g<1>10', worker)

    if re.search(r"max_iters\s*=\s*1\b", worker) or re.search(r'"max_iters"\s*:\s*1\b', worker):
        raise RuntimeError("One-step max_iters marker remains in generated ten-step worker")
    if not re.search(r"max_iters\s*=\s*10\b", worker):
        raise RuntimeError("Ten-step train-call max_iters marker missing")
    if not re.search(r'"max_iters"\s*:\s*10\b', worker):
        raise RuntimeError("Ten-step summary max_iters marker missing")
    return worker


def main() -> None:
    out = Path(".kaggle_strict_oof_ten_step_probe_kernel")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "run_ten_step_probe.py").write_text(build_worker(), encoding="utf-8")
    meta = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "run_ten_step_probe.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [SUPPORT_DATASET],
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({
        "kernel": KERNEL_ID,
        "train": PROBE_TRAIN,
        "validation": PROBE_VAL,
        "max_iters": 10,
        "github_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
    }, indent=2))


if __name__ == "__main__":
    main()
