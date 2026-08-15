from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_strict_oof_smoke_kernel import KERNEL_TITLE, TRAIN_NAMES, VAL_NAMES
from build_strict_oof_smoke_kernel_v2 import KERNEL_ID
from build_strict_oof_smoke_kernel_v3 import build_worker as build_worker_v3


def build_worker() -> str:
    worker = build_worker_v3()

    # Never recursively traverse /kaggle/input. OME-Zarr contains a very large
    # number of chunk files and a global rglob can trigger a hard Kaggle cancel.
    recursive_block = '''    # Also discover the parent of a known GT file if Kaggle changes the mount.\n    input_root = Path("/kaggle/input")\n    if input_root.exists():\n        for probe in input_root.rglob(f"{declared_names[0]}.geff"):\n            if probe.parent not in root_candidates:\n                root_candidates.append(probe.parent)\n\n'''
    if recursive_block not in worker:
        raise RuntimeError("Expected recursive input scan block not found")
    worker = worker.replace(recursive_block, "", 1)

    support_fallback = '''    root = Path("/kaggle/input")\n    if root.exists():\n        for manifest in root.rglob("ARTIFACT_MANIFEST.json"):\n            if "biohub-tracking-support-pack-50ep-v1" in str(manifest):\n                return manifest.parent\n    raise FileNotFoundError("Attached biohub-tracking-support-pack-50ep-v1 was not found")\n'''
    direct_failure = '''    raise FileNotFoundError("Attached biohub-tracking-support-pack-50ep-v1 was not found at known direct paths")\n'''
    if support_fallback not in worker:
        raise RuntimeError("Expected recursive support fallback not found")
    worker = worker.replace(support_fallback, direct_failure, 1)

    enter = '''    global TRAIN\n\n    # Resolve the competition train root at runtime.'''
    enter_repl = '''    global TRAIN\n    (OUT / "stage_00_enter_run.json").write_text(json.dumps({"status":"entered_run"}, indent=2), encoding="utf-8")\n\n    # Resolve the competition train root at runtime.'''
    if enter not in worker:
        raise RuntimeError("Could not locate run-entry marker")
    worker = worker.replace(enter, enter_repl, 1)

    support_block = '''    TRAIN = selected\n\n    support = find_support_root()\n    ensure_dependencies(support)\n    copy_vendored_official()\n'''
    support_repl = '''    TRAIN = selected\n    (OUT / "stage_01_input_ok.json").write_text(json.dumps({"train_root":str(TRAIN),"n_declared":len(declared_names)}, indent=2), encoding="utf-8")\n\n    (OUT / "stage_02_before_support.json").write_text(json.dumps({"status":"before_support_lookup"}, indent=2), encoding="utf-8")\n    support = next((p for p in SUPPORT_CANDIDATES if p.exists()), None)\n    if support is None:\n        raise FileNotFoundError("Support pack not found at known direct paths: " + ", ".join(str(p) for p in SUPPORT_CANDIDATES))\n    (OUT / "stage_03_support_ok.json").write_text(json.dumps({"support_root":str(support)}, indent=2), encoding="utf-8")\n\n    (OUT / "stage_04_before_dependencies.json").write_text(json.dumps({"status":"before_dependency_recovery"}, indent=2), encoding="utf-8")\n    ensure_dependencies(support)\n    (OUT / "stage_05_dependencies_ok.json").write_text(json.dumps({"status":"dependencies_ready"}, indent=2), encoding="utf-8")\n\n    copy_vendored_official()\n    (OUT / "stage_06_official_source_ok.json").write_text(json.dumps({"official":str(OFFICIAL)}, indent=2), encoding="utf-8")\n'''
    if support_block not in worker:
        raise RuntimeError("Could not locate support/dependency block")
    worker = worker.replace(support_block, support_repl, 1)

    imports_tail = '''    from train_unet_transformer import train\n\n    if not torch.cuda.is_available():'''
    imports_repl = '''    from train_unet_transformer import train\n    (OUT / "stage_07_imports_ok.json").write_text(json.dumps({"torch":str(torch.__version__),"cuda_available":bool(torch.cuda.is_available())}, indent=2), encoding="utf-8")\n\n    if not torch.cuda.is_available():'''
    if imports_tail not in worker:
        raise RuntimeError("Could not locate model-import marker")
    worker = worker.replace(imports_tail, imports_repl, 1)

    train_marker = '''    (OUT / "progress.json").write_text(json.dumps({"phase":"training"}, indent=2), encoding="utf-8")\n    t0 = time.time()\n    model = train('''
    train_repl = '''    (OUT / "progress.json").write_text(json.dumps({"phase":"training"}, indent=2), encoding="utf-8")\n    (OUT / "stage_08_before_train.json").write_text(json.dumps({"status":"about_to_call_train"}, indent=2), encoding="utf-8")\n    t0 = time.time()\n    model = train('''
    if train_marker not in worker:
        raise RuntimeError("Could not locate train marker")
    worker = worker.replace(train_marker, train_repl, 1)

    after_train = '''    train_seconds = time.time() - t0\n    model.to(device); model.eval()\n'''
    after_train_repl = '''    train_seconds = time.time() - t0\n    (OUT / "stage_09_train_returned.json").write_text(json.dumps({"train_seconds":train_seconds}, indent=2), encoding="utf-8")\n    model.to(device); model.eval()\n'''
    if after_train not in worker:
        raise RuntimeError("Could not locate post-train marker")
    worker = worker.replace(after_train, after_train_repl, 1)
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
        'mode': 'staged-no-global-rglob-v4',
    }, indent=2))


if __name__ == '__main__':
    main()
