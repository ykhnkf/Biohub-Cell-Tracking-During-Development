from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

KERNEL_ID = "ykhnkf/biohub-cell-tracking-cv-audit"
KERNEL_TITLE = "Biohub Cell Tracking CV Audit"
COMPETITION = "biohub-cell-tracking-during-development"
SUPPORT_DATASET = "pilkwang/biohub-tracking-support-pack-50ep-v1"
OFFICIAL_REPO = "https://github.com/royerlab/kaggle-cell-tracking-competition.git"
OFFICIAL_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"


def vendor_official_source(out: Path) -> None:
    """Vendor the exact official evaluator revision on the GitHub runner.

    Kaggle execution is intentionally offline.  We fetch source here, before
    pushing the kernel, so the Kaggle runtime cannot drift with upstream code or
    fail because of network access.
    """
    checkout = out / "official"
    if checkout.exists():
        shutil.rmtree(checkout)
    subprocess.run(["git", "clone", "--quiet", "--no-checkout", OFFICIAL_REPO, str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--quiet", OFFICIAL_COMMIT], check=True)
    git_dir = checkout / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)


def build_worker() -> str:
    return rf'''from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
import traceback
from pathlib import Path

OUT = Path("/kaggle/working/biohub_cv_audit")
OUT.mkdir(parents=True, exist_ok=True)
TRAIN = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train")
OFFICIAL = Path("/kaggle/working/official")
OFFICIAL_COMMIT = "{OFFICIAL_COMMIT}"
SUPPORT_CANDIDATES = [
    Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"),
    Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1"),
]

# Minimal dependency set for reading GEFF and running the official evaluator.
# The public learned-graph notebook ships these as offline wheels in its support
# dataset.  --no-deps avoids replacing Kaggle numpy/scipy with incompatible
# binary builds in the live kernel.
PACKAGE_SPECS = {{
    "tracksdata": "tracksdata",
    "geff": "geff>=1.1.3.1.1",
    "geff_spec": "geff-spec<1.2",
    "polars": "polars>=1.36",
    "zarr": "zarr>=3.0.10,<4",
    "bidict": "bidict>=0.23.1",
    "psygnal": "psygnal>=0.14",
    "rich": "rich",
    "networkx": "networkx>=3.2.1",
    "pydantic": "pydantic>=2.11",
    "pydantic_core": "pydantic-core",
    "annotated_types": "annotated-types",
    "typing_inspection": "typing-inspection",
    "markdown_it": "markdown-it-py",
    "pygments": "pygments",
    "numcodecs": "numcodecs>=0.13,<0.16",
    "donfig": "donfig>=0.8",
    "google_crc32c": "google-crc32c>=1.5",
    "deprecated": "deprecated",
    "wrapt": "wrapt",
    "msgpack": "msgpack",
}}


def module_missing(name: str) -> bool:
    return importlib.util.find_spec(name) is None


def find_support_root() -> Path:
    for p in SUPPORT_CANDIDATES:
        if p.exists():
            return p
    root = Path("/kaggle/input")
    if root.exists():
        for p in root.rglob("ARTIFACT_MANIFEST.json"):
            if "biohub-tracking-support-pack-50ep-v1" in str(p):
                return p.parent
    raise FileNotFoundError("Attached Biohub support dataset was not found")


def offline_dirs(support: Path) -> list[Path]:
    candidates = [support / "wheels", support]
    return [p for p in candidates if p.exists() and any(p.glob("*.whl"))]


def ensure_eval_dependencies(support: Path) -> None:
    # First try the native Kaggle environment.  Install only missing modules.
    for _ in range(4):
        missing = [name for name in PACKAGE_SPECS if module_missing(name)]
        if not missing:
            # Imports catch binary/transitive failures that find_spec misses.
            failures = {{}}
            for name in PACKAGE_SPECS:
                try:
                    importlib.import_module(name)
                except Exception as exc:
                    failures[name] = f"{{type(exc).__name__}}: {{exc}}"
            if not failures:
                return
            missing = list(failures)

        dirs = offline_dirs(support)
        if not dirs:
            raise RuntimeError(f"No offline wheel directory found under {{support}}; missing={{missing}}")
        specs = [PACKAGE_SPECS[name] for name in missing if name in PACKAGE_SPECS]
        cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
        for d in dirs:
            cmd.extend(["--find-links", str(d)])
        cmd.extend(specs)
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Offline dependency install failed\n"
                + (proc.stdout or "")[-4000:]
                + "\n"
                + (proc.stderr or "")[-4000:]
            )
        importlib.invalidate_caches()
        # Drop modules that may have been partially imported before install.
        for root in list(PACKAGE_SPECS):
            for key in list(sys.modules):
                if key == root or key.startswith(root + "."):
                    sys.modules.pop(key, None)
    raise RuntimeError("Offline dependency recovery did not converge")


def copy_vendored_official() -> None:
    # GitHub Actions uploads the vendored source beside this script.  Kaggle
    # mounts kernel source read-only under /kaggle/src; locate it robustly.
    here = Path(__file__).resolve().parent
    candidates = [here / "official", Path("/kaggle/src/official")]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(f"Vendored official source not found; checked {{candidates}}")
    import shutil
    if OFFICIAL.exists():
        shutil.rmtree(OFFICIAL)
    shutil.copytree(src, OFFICIAL)


def load_graph(td, path: Path):
    obj = td.graph.IndexedRXGraph.from_geff(path)
    return obj[0] if isinstance(obj, tuple) else obj


def run() -> None:
    support = find_support_root()
    (OUT / "dependency_probe.json").write_text(json.dumps({{
        "support_root": str(support),
        "wheel_dirs": [str(x) for x in offline_dirs(support)],
        "preinstalled": {{name: not module_missing(name) for name in PACKAGE_SPECS}},
    }}, indent=2), encoding="utf-8")

    ensure_eval_dependencies(support)
    copy_vendored_official()
    sys.path.insert(0, str(OFFICIAL / "scripts"))
    sys.path.insert(0, str(OFFICIAL / "src"))

    import tracksdata as td
    from geff import GeffMetadata
    from tracking_cellmot.io import DEFAULT_SCALE, open_dataset
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise

    def dataset_scale(name: str):
        try:
            return tuple(float(x) for x in open_dataset(TRAIN / name, load_image=False).scale)
        except Exception:
            return tuple(float(x) for x in DEFAULT_SCALE)

    def estimated_total(path: Path):
        try:
            meta = GeffMetadata.read(path)
            val = (meta.extra or {{}}).get("estimated_number_of_nodes")
            return float(val) if val is not None else float("nan")
        except Exception:
            return float("nan")

    def divisions(graph):
        ids = graph.node_ids()
        if len(ids) == 0:
            return 0
        return int(sum(int(x) >= 2 for x in graph.out_degree(ids)))

    geffs = sorted(TRAIN.glob("*.geff"))
    if not geffs:
        geffs = sorted(TRAIN.rglob("*.geff"))
    if not geffs:
        raise RuntimeError(f"No ground-truth .geff files found below {{TRAIN}}")

    rows = []
    metric_rows = []
    selftest_failures = []

    for path in geffs:
        name = path.stem
        graph = load_graph(td, path)
        n_nodes = int(graph.num_nodes())
        n_edges = int(graph.num_edges())
        n_div = divisions(graph)
        n_total = estimated_total(path)
        scale = dataset_scale(name)

        pred = load_graph(td, path)
        gt = load_graph(td, path)
        er = evaluate(pred, gt, scale=scale, max_distance=7.0)
        rec = node_recall(pred, gt) if n_nodes and n_edges else 0.0
        pm = per_sample_metrics(er, n_total, rec)
        metric_rows.append(pm)
        edge_j = float(pm["edge_jaccard"])
        if not (math.isnan(edge_j) or edge_j > 0.999999):
            selftest_failures.append({{"dataset": name, "edge_jaccard": edge_j}})

        rows.append({{
            "dataset": name,
            "gt_nodes": n_nodes,
            "gt_edges": n_edges,
            "gt_divisions": n_div,
            "estimated_number_of_nodes": n_total,
            "scale_z": scale[0],
            "scale_y": scale[1],
            "scale_x": scale[2],
            "self_edge_tp": int(er.edge_tp),
            "self_edge_fp": int(er.edge_fp),
            "self_edge_fn": int(er.edge_fn),
            "self_div_tp": int(er.division_tp),
            "self_div_fp": int(er.division_fp),
            "self_div_fn": int(er.division_fn),
            "self_node_recall": float(rec),
        }})

    with (OUT / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    names = [r["dataset"] for r in rows]
    lodo = [
        {{"fold": i, "train": [x for x in names if x != holdout], "valid": [holdout]}}
        for i, holdout in enumerate(names)
    ]
    (OUT / "folds_lodo.json").write_text(json.dumps(lodo, indent=2), encoding="utf-8")

    k = min(5, len(rows))
    folds = [{{"fold": i, "valid": [], "edges": 0, "divisions": 0, "nodes": 0}} for i in range(k)]
    for r in sorted(rows, key=lambda x: (x["gt_edges"], x["gt_divisions"], x["gt_nodes"]), reverse=True):
        target = min(folds, key=lambda f: (f["edges"], f["divisions"], f["nodes"], f["fold"]))
        target["valid"].append(r["dataset"])
        target["edges"] += r["gt_edges"]
        target["divisions"] += r["gt_divisions"]
        target["nodes"] += r["gt_nodes"]
    for f in folds:
        f["train"] = [x for x in names if x not in set(f["valid"])]
    (OUT / "folds_balanced_5.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")

    summary_metric = summarise(metric_rows)
    selftest = {{
        "datasets": len(rows),
        "official_commit": OFFICIAL_COMMIT,
        "summary": {{k: (float(v) if hasattr(v, "__float__") else v) for k, v in summary_metric.items()}},
        "failures": selftest_failures,
    }}
    (OUT / "official_metric_selftest.json").write_text(json.dumps(selftest, indent=2, default=str), encoding="utf-8")

    summary = {{
        "competition": "biohub-cell-tracking-during-development",
        "train_root": str(TRAIN),
        "support_root": str(support),
        "official_commit": OFFICIAL_COMMIT,
        "n_datasets": len(rows),
        "total_gt_nodes": sum(r["gt_nodes"] for r in rows),
        "total_gt_edges": sum(r["gt_edges"] for r in rows),
        "total_gt_divisions": sum(r["gt_divisions"] for r in rows),
        "primary_cv": "leave-one-dataset-out",
        "secondary_cv": f"balanced_{{k}}_fold",
        "metric_selftest_failures": len(selftest_failures),
    }}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "audit_environment.json").write_text(json.dumps({{
        "python": sys.version,
        "platform": platform.platform(),
        "official_commit": OFFICIAL_COMMIT,
        "tracksdata": getattr(td, "__version__", "unknown"),
    }}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


try:
    run()
except Exception as exc:
    payload = {{
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }}
    (OUT / "fatal_error.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise
'''


def main() -> None:
    out = Path(".kaggle_cv_audit_kernel")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    vendor_official_source(out)
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
        "out": str(out),
        "official_commit": OFFICIAL_COMMIT,
        "support_dataset": SUPPORT_DATASET,
    }, indent=2))


if __name__ == "__main__":
    main()
