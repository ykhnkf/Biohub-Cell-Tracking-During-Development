from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_classical_detection_pilot_kernel import KERNEL_ID, KERNEL_TITLE, build_worker as build_worker_v1


def build_worker() -> str:
    worker = build_worker_v1()
    old_import = '''    import numpy as np
    import zarr
    from scipy.ndimage import gaussian_filter, maximum_filter
'''
    new_import = '''    import blosc2
    import numpy as np
    from scipy.ndimage import gaussian_filter, maximum_filter
'''
    if old_import not in worker:
        raise RuntimeError("Expected zarr import block not found")
    worker = worker.replace(old_import, new_import, 1)

    old_open = '''        ds = open_dataset(TRAIN / name, normalize=False, load_image=False)
        scale = np.asarray(ds.scale, dtype=np.float64)
        arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
        if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
            raise ValueError(f"Missing 0.001/0.999 image quantiles for {name}")
        qlow = float(ds.quantiles["0.001"])
        qhigh = float(ds.quantiles["0.999"])
        T = int(ds.image_shape[0])
'''
    new_open = '''        ds = open_dataset(TRAIN / name, normalize=False, load_image=False)
        scale = np.asarray(ds.scale, dtype=np.float64)
        zarr_path = TRAIN / f"{name}.zarr"
        arr_meta = json.loads((zarr_path / "0" / "zarr.json").read_text(encoding="utf-8"))
        image_shape = tuple(int(x) for x in arr_meta["shape"])
        image_dtype = np.dtype(arr_meta["data_type"])
        if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
            raise ValueError(f"Missing 0.001/0.999 image quantiles for {name}")
        qlow = float(ds.quantiles["0.001"])
        qhigh = float(ds.quantiles["0.999"])
        T = int(image_shape[0])
'''
    if old_open not in worker:
        raise RuntimeError("Expected zarr open block not found")
    worker = worker.replace(old_open, new_open, 1)

    old_read = '''            raw = arr[t, ::downsample[0], ::downsample[1], ::downsample[2]].astype(np.float32)
            norm = (raw - qlow) / (qhigh - qlow + 1e-6)
'''
    new_read = '''            chunk_path = zarr_path / "0" / "c" / str(t) / "0" / "0" / "0"
            compressed = chunk_path.read_bytes()
            decompressed = blosc2.decompress(compressed)
            full = np.frombuffer(decompressed, dtype=image_dtype).reshape(image_shape[1:])
            raw = full[::downsample[0], ::downsample[1], ::downsample[2]].astype(np.float32)
            norm = (raw - qlow) / (qhigh - qlow + 1e-6)
'''
    if old_read not in worker:
        raise RuntimeError("Expected zarr frame read block not found")
    worker = worker.replace(old_read, new_read, 1)
    return worker


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
    print(json.dumps({"kernel": KERNEL_ID, "github_run_id": os.environ.get("GITHUB_RUN_ID", "local"), "reader": "blosc2-direct"}, indent=2))


if __name__ == "__main__":
    main()
