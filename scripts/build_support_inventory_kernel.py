from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

KERNEL_ID = "ykhnkf/biohub-support-pack-inventory"
KERNEL_TITLE = "Biohub Support Pack Inventory"
SUPPORT_DATASET = "pilkwang/biohub-tracking-support-pack-50ep-v1"

WORKER = r'''from __future__ import annotations
import json
import os
from pathlib import Path

OUT = Path("/kaggle/working/biohub_support_inventory")
OUT.mkdir(parents=True, exist_ok=True)
RUN_ID = __RUN_ID__
(OUT / "run_token.json").write_text(json.dumps({"github_run_id": RUN_ID}, indent=2), encoding="utf-8")

candidates = [
    Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"),
    Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1"),
]
support = next((p for p in candidates if p.exists()), None)
if support is None:
    for p in Path("/kaggle/input").rglob("ARTIFACT_MANIFEST.json"):
        if "biohub-tracking-support-pack-50ep-v1" in str(p):
            support = p.parent
            break
if support is None:
    raise FileNotFoundError("support pack not found")

files = []
json_docs = []
for path in sorted(p for p in support.rglob("*") if p.is_file()):
    rel = str(path.relative_to(support))
    item = {"path": rel, "size": path.stat().st_size, "suffix": path.suffix.lower()}
    files.append(item)
    if path.suffix.lower() == ".json" and path.stat().st_size <= 2_000_000:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                desc = {"type": "dict", "keys": list(obj.keys())[:100]}
                # Preserve compact scalar/list metadata that may reveal folds/splits.
                sample = {}
                for k, v in obj.items():
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        sample[k] = v
                    elif isinstance(v, list) and len(v) <= 50 and all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
                        sample[k] = v
                    elif isinstance(v, dict) and len(v) <= 20:
                        sample[k] = {str(kk): (vv if isinstance(vv, (str, int, float, bool)) or vv is None else f"<{type(vv).__name__} len={len(vv) if hasattr(vv, '__len__') else '?'}>") for kk, vv in list(v.items())[:20]}
                desc["sample"] = sample
            elif isinstance(obj, list):
                desc = {"type": "list", "length": len(obj), "first_types": [type(x).__name__ for x in obj[:10]]}
                if len(obj) <= 30:
                    desc["sample"] = obj
            else:
                desc = {"type": type(obj).__name__, "value": obj}
            json_docs.append({"path": rel, "description": desc})
        except Exception as exc:
            json_docs.append({"path": rel, "parse_error": f"{type(exc).__name__}: {exc}"})

weights = [f for f in files if f["suffix"] in {".pt", ".pth", ".ckpt", ".safetensors"}]
configs = [f for f in files if f["suffix"] in {".json", ".yaml", ".yml", ".toml", ".txt", ".md"}]
summary = {
    "support_root": str(support),
    "n_files": len(files),
    "total_bytes": sum(f["size"] for f in files),
    "n_weights": len(weights),
    "weights": weights,
    "config_like_files": configs,
    "json_documents": json_docs,
}
(OUT / "support_inventory.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
(OUT / "all_files.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2, default=str))
'''


def main() -> None:
    out = Path(".kaggle_support_inventory_kernel")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    worker = WORKER.replace("__RUN_ID__", repr(run_id))
    (out / "run_inventory.py").write_text(worker, encoding="utf-8")
    metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "run_inventory.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [SUPPORT_DATASET],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
