from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_strict_oof_smoke_kernel import KERNEL_TITLE, TRAIN_NAMES, VAL_NAMES
from build_strict_oof_smoke_kernel_v2 import KERNEL_ID
from build_strict_oof_smoke_kernel import build_worker as build_worker_v2base


def build_worker() -> str:
    worker = build_worker_v2base()
    needle = '''def run() -> None:\n    import gc\n    import time\n    support = find_support_root()\n'''
    replacement = '''def run() -> None:\n    import gc\n    import time\n    global TRAIN\n\n    # Resolve the competition train root at runtime. Kaggle may expose the\n    # competition source under different mount layouts across kernel types.\n    declared_names = __ALL_NAMES__\n    root_candidates = [\n        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),\n        Path("/kaggle/input/biohub-cell-tracking-during-development/train"),\n    ]\n    # Also discover the parent of a known GT file if Kaggle changes the mount.\n    input_root = Path("/kaggle/input")\n    if input_root.exists():\n        for probe in input_root.rglob(f"{declared_names[0]}.geff"):\n            if probe.parent not in root_candidates:\n                root_candidates.append(probe.parent)\n\n    root_diag = []\n    selected = None\n    for root in root_candidates:\n        paired = []\n        for name in declared_names:\n            paired.append({\n                "dataset": name,\n                "geff": (root / f"{name}.geff").exists(),\n                "zarr": (root / f"{name}.zarr").exists(),\n            })\n        n_pairs = sum(int(x["geff"] and x["zarr"]) for x in paired)\n        root_diag.append({\n            "root": str(root),\n            "exists": root.exists(),\n            "n_expected_pairs": n_pairs,\n            "n_declared": len(declared_names),\n            "pairs": paired,\n        })\n        if n_pairs == len(declared_names) and selected is None:\n            selected = root\n\n    (OUT / "strict_input_layout.json").write_text(json.dumps({\n        "initial_train": str(TRAIN),\n        "candidates": root_diag,\n        "selected": str(selected) if selected is not None else None,\n    }, indent=2), encoding="utf-8")\n    if selected is None:\n        raise FileNotFoundError("Could not locate a train root containing all declared GEFF/Zarr pairs; see strict_input_layout.json")\n    TRAIN = selected\n\n    support = find_support_root()\n'''.replace('__ALL_NAMES__', repr(TRAIN_NAMES + VAL_NAMES))
    if needle not in worker:
        raise RuntimeError('Could not locate strict OOF run preamble')
    worker = worker.replace(needle, replacement, 1)
    return worker


def main() -> None:
    out = Path('.kaggle_strict_oof_smoke_kernel')
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / 'run_strict_oof_smoke.py').write_text(build_worker(), encoding='utf-8')
    meta = {
        'id': KERNEL_ID,
        'title': KERNEL_TITLE,
        'code_file': 'run_strict_oof_smoke.py',
        'language': 'python',
        'kernel_type': 'script',
        'is_private': True,
        'enable_gpu': True,
        'enable_tpu': False,
        'enable_internet': False,
        'keywords': [],
        'dataset_sources': [SUPPORT_DATASET],
        'kernel_sources': [],
        'competition_sources': [COMPETITION],
        'model_sources': [],
    }
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps({
        'kernel': KERNEL_ID,
        'github_run_id': os.environ.get('GITHUB_RUN_ID', 'local'),
        'n_train': len(TRAIN_NAMES),
        'n_val': len(VAL_NAMES),
        'mode': 'runtime-train-root-resolution-v3',
    }, indent=2))


if __name__ == '__main__':
    main()
