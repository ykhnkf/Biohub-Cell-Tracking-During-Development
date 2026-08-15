from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_cv_audit_kernel_v7 import build_worker as build_audit_worker_v7
from build_strict_oof_smoke_kernel import TRAIN_NAMES

KERNEL_ID = "ykhnkf/biohub-strict-oof-loader-probe"
KERNEL_TITLE = "Biohub Strict OOF Loader Probe"
PROBE_DATASET = TRAIN_NAMES[0]


def build_worker() -> str:
    base = build_audit_worker_v7()
    start = base.index("def run() -> None:")
    prefix = base[:start].replace(
        'OUT = Path("/kaggle/working/biohub_cv_audit")',
        'OUT = Path("/kaggle/working/biohub_strict_oof_loader_probe")',
        1,
    )
    dataset = repr(PROBE_DATASET)
    run = r'''def run() -> None:
    import gc
    import torch

    name = __DATASET__
    train = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train")
    (OUT / "stage_00_started.json").write_text(json.dumps({
        "dataset": name, "train_root": str(train), "train_exists": train.exists(),
    }, indent=2), encoding="utf-8")

    geff = train / f"{name}.geff"
    zarr_path = train / f"{name}.zarr"
    layout = {"geff": geff.exists(), "zarr": zarr_path.exists()}
    (OUT / "stage_01_paths.json").write_text(json.dumps(layout, indent=2), encoding="utf-8")
    if not all(layout.values()):
        raise FileNotFoundError(f"Probe dataset pair missing: {layout}")

    support = next((p for p in SUPPORT_CANDIDATES if p.exists()), None)
    if support is None:
        raise FileNotFoundError("Support pack missing at known direct paths")
    (OUT / "stage_02_support.json").write_text(json.dumps({"support": str(support)}, indent=2), encoding="utf-8")

    (OUT / "stage_03_before_dependencies.json").write_text(json.dumps({"status":"before"}, indent=2), encoding="utf-8")
    ensure_dependencies(support)
    (OUT / "stage_04_dependencies_ok.json").write_text(json.dumps({"status":"ok"}, indent=2), encoding="utf-8")

    copy_vendored_official()
    sys.path.insert(0, str(OFFICIAL / "scripts"))
    sys.path.insert(0, str(OFFICIAL / "src"))
    (OUT / "stage_05_source_ok.json").write_text(json.dumps({"official": str(OFFICIAL)}, indent=2), encoding="utf-8")

    from train_unet_transformer import FrameWindowDataset, load_dataset_windows
    (OUT / "stage_06_imports_ok.json").write_text(json.dumps({
        "torch": str(torch.__version__), "cuda_available": bool(torch.cuda.is_available())
    }, indent=2), encoding="utf-8")

    (OUT / "stage_07_before_window_metadata.json").write_text(json.dumps({"status":"before"}, indent=2), encoding="utf-8")
    vm, windows = load_dataset_windows(
        train / name,
        window_size=2,
        invert_time=False,
        max_frames=None,
        downsample=(1, 4, 4),
    )
    window_summary = {
        "zarr_path": str(vm.zarr_path),
        "image_shape": list(vm.image_shape),
        "downsample": list(vm.downsample),
        "voxel_size": [float(x) for x in vm.voxel_size],
        "q_low": float(vm.q_low),
        "q_high": float(vm.q_high),
        "n_windows": len(windows),
        "first_t_start": int(windows[0].t_start) if windows else None,
        "first_node_counts": list(windows[0].node_counts) if windows else None,
    }
    (OUT / "stage_08_window_metadata_ok.json").write_text(json.dumps(window_summary, indent=2), encoding="utf-8")
    if not windows:
        raise RuntimeError("No valid 2-frame annotated windows in probe dataset")

    ds = FrameWindowDataset([(vm, [windows[0]])], augmentations=[])
    (OUT / "stage_09_before_getitem.json").write_text(json.dumps({"len":len(ds),"max_nodes":int(ds.max_nodes)}, indent=2), encoding="utf-8")
    sample = ds[0]

    def tensor_info(x):
        return {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
            "finite": bool(torch.isfinite(x.float()).all().item()) if x.numel() else True,
        }

    sample_summary = {
        key: tensor_info(value)
        for key, value in sample.items()
        if isinstance(value, torch.Tensor)
    }
    sample_summary["imgs_min"] = float(sample["imgs"].float().min().item())
    sample_summary["imgs_max"] = float(sample["imgs"].float().max().item())
    sample_summary["status"] = "passed"
    (OUT / "loader_probe_summary.json").write_text(json.dumps(sample_summary, indent=2), encoding="utf-8")
    (OUT / "stage_10_getitem_ok.json").write_text(json.dumps({"status":"ok"}, indent=2), encoding="utf-8")
    print(json.dumps({"dataset":name,"window":window_summary,"sample":sample_summary}, indent=2))
    del sample, ds, windows, vm
    gc.collect()

try:
    run()
except Exception as exc:
    payload = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    (OUT / "fatal_error.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise
'''
    return prefix + run.replace("__DATASET__", dataset)


def main() -> None:
    out = Path(".kaggle_strict_oof_loader_probe_kernel")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "run_loader_probe.py").write_text(build_worker(), encoding="utf-8")
    meta = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "run_loader_probe.py",
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
    print(json.dumps({"kernel":KERNEL_ID,"dataset":PROBE_DATASET,"github_run_id":os.environ.get("GITHUB_RUN_ID","local")}, indent=2))


if __name__ == "__main__":
    main()
