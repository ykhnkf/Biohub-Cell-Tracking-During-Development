from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_classical_detection_pilot_kernel import KERNEL_ID, KERNEL_TITLE
from build_classical_detection_pilot_kernel_v2 import build_worker as build_worker_v2


def build_worker() -> str:
    worker = build_worker_v2()
    old = '''        nodes = [\n            {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}\n            for t, z, y, x in coords\n        ]\n'''
    new = '''        # The GT GEFF schema stores t/z/y/x as integer columns.  Because we\n        # preserve that schema when clearing the graph, predicted coordinates\n        # must also be native ints; passing 0.0/4.0 etc. causes recent\n        # tracksdata+Polars to reject Float64 values for an Int64 Series during\n        # official matching/evaluation.  These classical detections already lie\n        # exactly on the original voxel grid, so integer conversion is lossless.\n        nodes = [\n            {"t": int(t), "z": int(round(z)), "y": int(round(y)), "x": int(round(x))}\n            for t, z, y, x in coords\n        ]\n'''
    if old not in worker:
        raise RuntimeError("Expected predicted-node construction block not found")
    return worker.replace(old, new, 1)


def main() -> None:
    out = Path(".kaggle_classical_detection_pilot_kernel")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "run_detection_pilot.py").write_text(build_worker(), encoding="utf-8")
    metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "run_detection_pilot.py",
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
        "github_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "reader": "blosc2-direct",
        "coord_schema_fix": "int-native",
    }, indent=2))


if __name__ == "__main__":
    main()
