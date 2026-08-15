from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_strict_oof_smoke_kernel import KERNEL_TITLE, TRAIN_NAMES, VAL_NAMES, build_worker

KERNEL_ID = "ykhnkf/biohub-strict-oof-train-predict-score-smoke"


def main() -> None:
    out = Path(".kaggle_strict_oof_smoke_kernel")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "run_strict_oof_smoke.py").write_text(build_worker(), encoding="utf-8")
    meta = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "run_strict_oof_smoke.py",
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
        "github_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "n_train": len(TRAIN_NAMES),
        "n_val": len(VAL_NAMES),
    }, indent=2))


if __name__ == "__main__":
    main()
