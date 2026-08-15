from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_cv_audit_kernel_v7 import build_worker as build_audit_worker_v7
from build_strict_oof_smoke_kernel import VAL_NAMES

KERNEL_ID = "ykhnkf/biohub-strict-oof-val-inventory"
KERNEL_TITLE = "Biohub Strict OOF Validation Inventory"


def build_worker() -> str:
    base = build_audit_worker_v7()
    start = base.index("def run() -> None:")
    prefix = base[:start].replace(
        'OUT = Path("/kaggle/working/biohub_cv_audit")',
        'OUT = Path("/kaggle/working/biohub_strict_oof_val_inventory")', 1,
    )
    vals = repr(VAL_NAMES)
    run = r'''def run() -> None:
    import numpy as np

    names = __VAL_NAMES__
    roots = [
        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
        Path("/kaggle/input/biohub-cell-tracking-during-development/train"),
    ]
    train = next((r for r in roots if all((r/f"{n}.geff").exists() and (r/f"{n}.zarr").exists() for n in names)), None)
    if train is None:
        raise FileNotFoundError("Could not resolve validation train root")
    support = next((p for p in SUPPORT_CANDIDATES if p.exists()), None)
    if support is None:
        raise FileNotFoundError("Support pack missing")
    ensure_dependencies(support)
    copy_vendored_official()
    sys.path.insert(0, str(OFFICIAL / "scripts"))
    sys.path.insert(0, str(OFFICIAL / "src"))
    from train_unet_transformer import load_dataset_windows

    rows=[]
    for name in names:
        vm, windows = load_dataset_windows(train/name, window_size=2, downsample=(1,4,4))
        counts=np.array([max(w.node_counts) for w in windows], dtype=np.int64)
        row={
            "dataset":name,
            "n_windows":int(len(windows)),
            "image_shape":list(vm.image_shape),
            "max_nodes":int(counts.max()) if len(counts) else 0,
            "node_p50":float(np.percentile(counts,50)) if len(counts) else 0.0,
            "node_p90":float(np.percentile(counts,90)) if len(counts) else 0.0,
            "node_p99":float(np.percentile(counts,99)) if len(counts) else 0.0,
            "q_low":float(vm.q_low),"q_high":float(vm.q_high),
        }
        rows.append(row)
        (OUT/f"inventory_{name}.json").write_text(json.dumps(row,indent=2),encoding="utf-8")
    payload={"status":"passed","train_root":str(train),"total_windows":sum(r["n_windows"] for r in rows),"datasets":rows}
    (OUT/"val_inventory.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))

try:
    run()
except Exception as exc:
    payload={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
    (OUT/"fatal_error.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2)); raise
'''
    return prefix + run.replace("__VAL_NAMES__", vals)


def main() -> None:
    out=Path(".kaggle_strict_oof_val_inventory_kernel")
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    (out/"run_val_inventory.py").write_text(build_worker(),encoding="utf-8")
    meta={"id":KERNEL_ID,"title":KERNEL_TITLE,"code_file":"run_val_inventory.py","language":"python","kernel_type":"script","is_private":True,"enable_gpu":False,"enable_tpu":False,"enable_internet":False,"keywords":[],"dataset_sources":[SUPPORT_DATASET],"kernel_sources":[],"competition_sources":[COMPETITION],"model_sources":[]}
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps({"kernel":KERNEL_ID,"validation":VAL_NAMES,"github_run_id":os.environ.get("GITHUB_RUN_ID","local")},indent=2))

if __name__=="__main__": main()
