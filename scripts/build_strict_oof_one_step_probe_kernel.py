from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_cv_audit_kernel_v7 import build_worker as build_audit_worker_v7
from build_strict_oof_smoke_kernel import TRAIN_NAMES, VAL_NAMES

KERNEL_ID = "ykhnkf/biohub-strict-oof-one-step-probe"
KERNEL_TITLE = "Biohub Strict OOF One Step Probe"
PROBE_TRAIN = TRAIN_NAMES[:2]
PROBE_VAL = VAL_NAMES[:1]


def build_worker() -> str:
    base = build_audit_worker_v7()
    start = base.index("def run() -> None:")
    prefix = base[:start].replace(
        'OUT = Path("/kaggle/working/biohub_cv_audit")',
        'OUT = Path("/kaggle/working/biohub_strict_oof_one_step_probe")',
        1,
    )
    support_fallback = '''    root = Path("/kaggle/input")\n    if root.exists():\n        for manifest in root.rglob("ARTIFACT_MANIFEST.json"):\n            if "biohub-tracking-support-pack-50ep-v1" in str(manifest):\n                return manifest.parent\n    raise FileNotFoundError("Attached biohub-tracking-support-pack-50ep-v1 was not found")\n'''
    direct_failure = '''    raise FileNotFoundError("Attached biohub-tracking-support-pack-50ep-v1 was not found at known direct paths")\n'''
    if support_fallback not in prefix:
        raise RuntimeError("Expected recursive support fallback not found in worker prefix")
    prefix = prefix.replace(support_fallback, direct_failure, 1)

    train_names = repr(PROBE_TRAIN)
    val_names = repr(PROBE_VAL)
    run = r'''def run() -> None:
    import gc
    import time
    import torch

    train_root = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train")
    train_names = __TRAIN_NAMES__
    val_names = __VAL_NAMES__
    (OUT / "stage_00_started.json").write_text(json.dumps({
        "train_root":str(train_root),"train":train_names,"validation":val_names
    }, indent=2), encoding="utf-8")

    overlap = sorted(set(train_names) & set(val_names))
    if overlap:
        raise RuntimeError(f"Probe train/val overlap: {overlap}")
    pairs = {
        n: {"geff":(train_root/f"{n}.geff").exists(),"zarr":(train_root/f"{n}.zarr").exists()}
        for n in train_names + val_names
    }
    (OUT / "stage_01_paths.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    if not all(v["geff"] and v["zarr"] for v in pairs.values()):
        raise FileNotFoundError(f"Missing probe pair: {pairs}")

    support = next((p for p in SUPPORT_CANDIDATES if p.exists()), None)
    if support is None:
        raise FileNotFoundError("Support pack missing at known direct paths")
    (OUT / "stage_02_support.json").write_text(json.dumps({"support":str(support)}, indent=2), encoding="utf-8")

    (OUT / "stage_03_before_dependencies.json").write_text(json.dumps({"status":"before"}, indent=2), encoding="utf-8")
    ensure_dependencies(support)
    (OUT / "stage_04_dependencies_ok.json").write_text(json.dumps({"status":"ok"}, indent=2), encoding="utf-8")

    copy_vendored_official()
    sys.path.insert(0, str(OFFICIAL / "scripts"))
    sys.path.insert(0, str(OFFICIAL / "src"))
    from train_unet_transformer import train
    (OUT / "stage_05_imports_ok.json").write_text(json.dumps({
        "torch":str(torch.__version__),"cuda_available":bool(torch.cuda.is_available()),
        "device_count":int(torch.cuda.device_count())
    }, indent=2), encoding="utf-8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gpu_before = {
        "name":torch.cuda.get_device_name(0),
        "allocated":int(torch.cuda.memory_allocated(device)),
        "reserved":int(torch.cuda.memory_reserved(device)),
    }
    (OUT / "stage_06_cuda_ready.json").write_text(json.dumps(gpu_before, indent=2), encoding="utf-8")

    split_path = OUT / "one_step_splits.json"
    split_path.write_text(json.dumps([
        {"split":0,"train":train_names,"test":val_names}
    ], indent=2), encoding="utf-8")
    (OUT / "stage_07_split_ready.json").write_text(json.dumps({
        "path":str(split_path),"train":train_names,"validation":val_names
    }, indent=2), encoding="utf-8")

    (OUT / "stage_08_before_train.json").write_text(json.dumps({
        "max_iters":1,"num_workers":0,"batch_size":1,"downsample":[1,4,4],
        "unet_layers":[32,64,128],"window_size":2
    }, indent=2), encoding="utf-8")
    t0 = time.time()
    model = train(
        data_dir=train_root,
        fold=0,
        splits_file=split_path,
        method="strict_oof_one_step_probe",
        n_epochs=1,
        lr=1e-4,
        batch_size=1,
        num_workers=0,
        unet_out_channels=32,
        unet_layers=[32,64,128],
        unet_weights=None,
        downsample=(1,4,4),
        det_loss_weight=1.0,
        det_neg_weight=1e-2,
        max_iters=1,
        seed=20260815,
        window_size=2,
        pool_kernel_um=5.0,
        data_parallel=False,
    )
    elapsed = time.time() - t0
    peak = int(torch.cuda.max_memory_allocated(device))
    reserved_peak = int(torch.cuda.max_memory_reserved(device))
    (OUT / "stage_09_train_returned.json").write_text(json.dumps({
        "seconds":elapsed,"peak_memory_allocated":peak,"peak_memory_reserved":reserved_peak
    }, indent=2), encoding="utf-8")

    model.to(device)
    model.eval()
    weights_dir = OFFICIAL / "weights" / "strict_oof_one_step_probe" / "split_0"
    weight_files = [str(p) for p in sorted(weights_dir.glob("*")) if p.is_file()]
    summary = {
        "status":"passed",
        "train":train_names,"validation":val_names,"overlap":overlap,
        "seconds":elapsed,"max_iters":1,"num_workers":0,"batch_size":1,
        "unet_layers":[32,64,128],"downsample":[1,4,4],"window_size":2,
        "gpu":gpu_before,
        "peak_memory_allocated":peak,"peak_memory_reserved":reserved_peak,
        "weight_files":weight_files,
    }
    (OUT / "one_step_probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()

try:
    run()
except Exception as exc:
    payload = {"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
    try:
        import torch
        if torch.cuda.is_available():
            payload["cuda"] = {
                "allocated":int(torch.cuda.memory_allocated()),
                "reserved":int(torch.cuda.memory_reserved()),
                "max_allocated":int(torch.cuda.max_memory_allocated()),
                "max_reserved":int(torch.cuda.max_memory_reserved()),
            }
    except Exception:
        pass
    (OUT / "fatal_error.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise
'''
    return prefix + run.replace("__TRAIN_NAMES__", train_names).replace("__VAL_NAMES__", val_names)


def main() -> None:
    out = Path(".kaggle_strict_oof_one_step_probe_kernel")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "run_one_step_probe.py").write_text(build_worker(), encoding="utf-8")
    meta = {
        "id":KERNEL_ID,"title":KERNEL_TITLE,"code_file":"run_one_step_probe.py",
        "language":"python","kernel_type":"script","is_private":True,
        "enable_gpu":True,"enable_tpu":False,"enable_internet":False,"keywords":[],
        "dataset_sources":[SUPPORT_DATASET],"kernel_sources":[],
        "competition_sources":[COMPETITION],"model_sources":[],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({
        "kernel":KERNEL_ID,"train":PROBE_TRAIN,"validation":PROBE_VAL,
        "github_run_id":os.environ.get("GITHUB_RUN_ID","local")
    }, indent=2))


if __name__ == "__main__":
    main()
