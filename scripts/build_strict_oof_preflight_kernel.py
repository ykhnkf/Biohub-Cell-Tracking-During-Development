from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_strict_oof_smoke_kernel import TRAIN_NAMES, VAL_NAMES

KERNEL_ID = "ykhnkf/biohub-strict-oof-gpu-preflight"
KERNEL_TITLE = "Biohub Strict OOF GPU Preflight"


def build_worker() -> str:
    all_names = repr(TRAIN_NAMES + VAL_NAMES)
    run_id = repr(os.environ.get("GITHUB_RUN_ID", "local"))
    return f'''from __future__ import annotations
import json
from pathlib import Path

OUT = Path("/kaggle/working/biohub_strict_oof_preflight")
OUT.mkdir(parents=True, exist_ok=True)

def write(name, payload):
    (OUT / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

write("stage_00_started.json", {{"github_run_id": {run_id}}})

# Stage A: CUDA only.
import torch
cuda = {{
    "available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()),
    "torch_version": str(torch.__version__),
    "cuda_version": str(torch.version.cuda),
}}
if torch.cuda.is_available():
    cuda["device_name"] = torch.cuda.get_device_name(0)
write("stage_01_cuda.json", cuda)
if not cuda["available"]:
    raise RuntimeError("CUDA unavailable")

# Stage B: direct competition mount only. No recursive /kaggle/input scan.
train = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train")
names = {all_names}
pairs = []
for name in names:
    pairs.append({{
        "dataset": name,
        "geff": (train / f"{{name}}.geff").exists(),
        "zarr": (train / f"{{name}}.zarr").exists(),
    }})
layout = {{
    "train_root": str(train),
    "train_exists": train.exists(),
    "top_level_count": len(list(train.iterdir())) if train.exists() else 0,
    "n_declared": len(names),
    "n_complete_pairs": sum(int(x["geff"] and x["zarr"]) for x in pairs),
    "pairs": pairs,
}}
write("stage_02_input_layout.json", layout)
if layout["n_complete_pairs"] != len(names):
    raise FileNotFoundError("Direct competition mount does not contain all declared pairs")

# Stage C: direct support-pack paths only. Again, no recursive scan.
support_candidates = [
    Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"),
    Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1"),
]
support = next((p for p in support_candidates if p.exists()), None)
support_diag = {{
    "candidates": [{{"path": str(p), "exists": p.exists()}} for p in support_candidates],
    "selected": str(support) if support is not None else None,
}}
write("stage_03_support.json", support_diag)
if support is None:
    raise FileNotFoundError("Support pack not found at direct known paths")

write("preflight_summary.json", {{
    "status": "passed",
    "github_run_id": {run_id},
    "cuda": cuda,
    "train_root": str(train),
    "n_complete_pairs": layout["n_complete_pairs"],
    "support_root": str(support),
}})
print(json.dumps({{"status":"passed","pairs":layout["n_complete_pairs"],"support":str(support),"cuda":cuda}}, indent=2))
'''


def main() -> None:
    out = Path(".kaggle_strict_oof_preflight_kernel")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "run_preflight.py").write_text(build_worker(), encoding="utf-8")
    meta = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "run_preflight.py",
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
    print(json.dumps({"kernel": KERNEL_ID, "mode": "gpu-preflight-no-recursion", "github_run_id": os.environ.get("GITHUB_RUN_ID", "local")}, indent=2))


if __name__ == "__main__":
    main()
