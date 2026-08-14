from __future__ import annotations

import json
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import (
    COMPETITION,
    KERNEL_ID,
    KERNEL_TITLE,
    OFFICIAL_COMMIT,
    SUPPORT_DATASET,
    build_worker as build_worker_v3,
)


def build_worker() -> str:
    worker = build_worker_v3()
    old = '''def copy_vendored_official() -> None:\n    here = Path(__file__).resolve().parent\n    src = next((p for p in [here / "official", Path("/kaggle/src/official")] if p.exists()), None)\n    if src is None:\n        raise FileNotFoundError("Vendored official source not found")\n    if OFFICIAL.exists():\n        shutil.rmtree(OFFICIAL)\n    shutil.copytree(src, OFFICIAL)\n'''
    new = '''def copy_vendored_official() -> None:\n    # The public learned-graph notebooks materialize the full baseline repo from\n    # the attached support pack. Reuse that exact packaging path instead of\n    # relying on auxiliary files beside a Kaggle script kernel.\n    support = find_support_root()\n    src_dir = support / "repo"\n    src_zip = support / "repo.zip"\n    if OFFICIAL.exists():\n        shutil.rmtree(OFFICIAL)\n    if src_dir.exists() and src_dir.is_dir():\n        shutil.copytree(src_dir, OFFICIAL)\n    elif src_zip.exists() and src_zip.is_file():\n        import zipfile\n        OFFICIAL.mkdir(parents=True, exist_ok=True)\n        with zipfile.ZipFile(src_zip) as zf:\n            zf.extractall(OFFICIAL)\n    else:\n        raise FileNotFoundError(f"Support pack has neither repo/ nor repo.zip under {support}")\n\n    required = [\n        OFFICIAL / "scripts" / "evaluate.py",\n        OFFICIAL / "src" / "tracking_cellmot" / "metrics.py",\n    ]\n    missing = [str(p) for p in required if not p.exists()]\n    if missing:\n        raise FileNotFoundError("Materialized support repo is missing evaluator files: " + ", ".join(missing))\n'''
    if old not in worker:
        raise RuntimeError("Could not locate v3 copy_vendored_official block")
    return worker.replace(old, new, 1)


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
        "evaluator_source": "support-pack repo/repo.zip",
        "reference_official_commit": OFFICIAL_COMMIT,
    }, indent=2))


if __name__ == "__main__":
    main()
