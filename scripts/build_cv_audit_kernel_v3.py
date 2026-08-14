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
    checkout = out / "official"
    if checkout.exists():
        shutil.rmtree(checkout)
    subprocess.run(["git", "clone", "--quiet", "--no-checkout", OFFICIAL_REPO, str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--quiet", OFFICIAL_COMMIT], check=True)
    if (checkout / ".git").exists():
        shutil.rmtree(checkout / ".git")


def build_worker() -> str:
    return rf'''from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import math
import platform
import re
import shutil
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

# This dependency map mirrors the public learned-graph notebook's tested
# offline bootstrap.  It deliberately excludes numpy/scipy/torch so pip cannot
# replace Kaggle's binary stack.
PACKAGE_SPECS = {{
    "tracksdata": ("tracksdata", "tracksdata"),
    "zarr": ("zarr", "zarr>=3.0.10,<4"),
    "pyscipopt": ("pyscipopt", "pyscipopt"),
    "geff": ("geff", "geff>=1.1.3.1.1"),
    "geff_spec": ("geff_spec", "geff-spec<1.2"),
    "ilpy": ("ilpy", "ilpy>=0.5.1"),
    "polars": ("polars", "polars>=1.36"),
    "blosc2": ("blosc2", "blosc2"),
    "dask": ("dask", "dask"),
    "imagecodecs": ("imagecodecs", "imagecodecs"),
    "skimage": ("skimage", "scikit-image>=0.24"),
    "pyarrow": ("pyarrow", "pyarrow"),
    "rustworkx": ("rustworkx", "rustworkx>=0.17.1"),
    "sqlalchemy": ("sqlalchemy", "sqlalchemy>=2"),
    "numcodecs": ("numcodecs", "numcodecs>=0.13,<0.16"),
    "donfig": ("donfig", "donfig>=0.8"),
    "google_crc32c": ("google_crc32c", "google-crc32c>=1.5"),
    "bidict": ("bidict", "bidict>=0.23.1"),
    "psygnal": ("psygnal", "psygnal>=0.14"),
    "rich": ("rich", "rich"),
    "networkx": ("networkx", "networkx>=3.2.1"),
    "pydantic": ("pydantic", "pydantic>=2.11"),
    "pydantic_core": ("pydantic_core", "pydantic-core"),
    "annotated_types": ("annotated_types", "annotated-types"),
    "typing_extensions": ("typing_extensions", "typing-extensions>=4.13"),
    "typing_inspection": ("typing_inspection", "typing-inspection"),
    "markdown_it": ("markdown_it", "markdown-it-py"),
    "pygments": ("pygments", "pygments"),
    "click": ("click", "click"),
    "cloudpickle": ("cloudpickle", "cloudpickle"),
    "fsspec": ("fsspec", "fsspec"),
    "partd": ("partd", "partd"),
    "locket": ("locket", "locket"),
    "toolz": ("toolz", "toolz"),
    "yaml": ("yaml", "pyyaml"),
    "ndindex": ("ndindex", "ndindex"),
    "msgpack": ("msgpack", "msgpack"),
    "numexpr": ("numexpr", "numexpr"),
    "deprecated": ("deprecated", "deprecated"),
    "wrapt": ("wrapt", "wrapt"),
    "imageio": ("imageio", "imageio"),
    "PIL": ("PIL", "pillow"),
    "tifffile": ("tifffile", "tifffile"),
    "lazy_loader": ("lazy_loader", "lazy-loader"),
    "tqdm": ("tqdm", "tqdm"),
}}
EXTRA_SPECS_BY_NAME = {{
    "tracksdata": ["bidict>=0.23.1", "psygnal>=0.14", "rich"],
    "zarr": ["donfig>=0.8", "google-crc32c>=1.5", "numcodecs>=0.13,<0.16"],
    "geff": ["geff-spec<1.2", "networkx>=3.2.1", "pydantic>=2.11", "numcodecs>=0.13,<0.16"],
    "geff_spec": ["pydantic>=2.11", "annotated-types", "pydantic-core", "typing-inspection"],
    "polars": ["polars-runtime-32"],
    "dask": ["click", "cloudpickle", "fsspec", "partd", "pyyaml", "toolz"],
    "partd": ["locket"],
    "blosc2": ["ndindex", "msgpack", "numexpr"],
    "numcodecs": ["deprecated", "msgpack", "wrapt"],
    "rich": ["markdown-it-py", "pygments"],
    "pydantic": ["annotated-types", "pydantic-core", "typing-extensions>=4.13", "typing-inspection"],
    "skimage": ["imageio", "pillow", "tifffile", "lazy-loader", "networkx"],
}}
REQUIRED_MODULES = {{name: module for name, (module, _) in PACKAGE_SPECS.items()}}


def module_missing(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is None


def find_support_root() -> Path:
    for path in SUPPORT_CANDIDATES:
        if path.exists():
            return path
    root = Path("/kaggle/input")
    if root.exists():
        for manifest in root.rglob("ARTIFACT_MANIFEST.json"):
            if "biohub-tracking-support-pack-50ep-v1" in str(manifest):
                return manifest.parent
    raise FileNotFoundError("Attached biohub-tracking-support-pack-50ep-v1 was not found")


def find_offline_package_dirs(support: Path) -> list[Path]:
    candidates = [support / "wheels", support, Path("/kaggle/working/wheels")]
    out = []
    for path in candidates:
        if path.exists() and path.is_dir() and any(path.glob("*.whl")):
            out.append(path)
    return out


def purge_imported_modules(package_names: list[str]) -> None:
    roots = set()
    for name in package_names:
        if name in PACKAGE_SPECS:
            roots.add(PACKAGE_SPECS[name][0].split(".")[0])
    roots.add("tracksdata")
    for root in roots:
        for module_name in list(sys.modules):
            if module_name == root or module_name.startswith(root + "."):
                sys.modules.pop(module_name, None)


def polars_runtime_ready() -> bool:
    try:
        import polars as pl
        from polars._plr import PySeries
        _ = PySeries
        return hasattr(pl, "Float16") and pl.Series([-999999.0], dtype=pl.Float64).dtype == pl.Float64
    except Exception:
        return False


def packages_requiring_refresh() -> list[str]:
    refresh = []
    if not module_missing("polars") and not polars_runtime_ready():
        refresh.append("polars")
    if not module_missing("zarr"):
        try:
            import zarr
            if int(str(getattr(zarr, "__version__", "0")).split(".", 1)[0]) < 3:
                refresh.append("zarr")
        except Exception:
            refresh.append("zarr")
    return refresh


def dependency_specs_for(missing: list[str]) -> list[str]:
    specs = []
    seen = set()
    for name in missing:
        candidates = []
        if name in PACKAGE_SPECS:
            candidates.append(PACKAGE_SPECS[name][1])
        candidates.extend(EXTRA_SPECS_BY_NAME.get(name, []))
        for spec in candidates:
            key = spec.lower()
            if key not in seen:
                seen.add(key)
                specs.append(spec)
    return specs


def import_failures() -> dict[str, str]:
    failures = {{}}
    for name, module_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures[name] = f"{{type(exc).__name__}}: {{exc}}"
    return failures


def missing_names_from_failures(failures: dict[str, str]) -> list[str]:
    names = []
    module_to_name = {{module: name for name, module in REQUIRED_MODULES.items()}}
    for message in failures.values():
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", message)
        if match:
            module = match.group(1).split(".")[0]
        else:
            match = re.search(r"module ['\"]([^'\"]+)['\"] has no attribute", message)
            if not match:
                continue
            module = match.group(1).split(".")[0]
        name = module_to_name.get(module)
        if name and name not in names:
            names.append(name)
    return names


def install_missing_dependencies(missing: list[str], support: Path) -> None:
    specs = dependency_specs_for(missing)
    if not specs:
        return
    dirs = find_offline_package_dirs(support)
    if not dirs:
        raise RuntimeError(f"No offline package dirs under {{support}}")
    cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
    if {{"polars", "zarr"}} & set(missing):
        cmd.append("--force-reinstall")
    for d in dirs:
        cmd.extend(["--find-links", str(d)])
    cmd.extend(specs)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    (OUT / "last_pip.json").write_text(json.dumps({{
        "missing": missing,
        "specs": specs,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-5000:],
        "stderr_tail": (proc.stderr or "")[-5000:],
    }}, indent=2), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError("Offline dependency install failed; see last_pip.json")
    purge_imported_modules(missing)
    importlib.invalidate_caches()


def ensure_dependencies(support: Path) -> None:
    for _ in range(6):
        refresh = packages_requiring_refresh()
        if refresh:
            install_missing_dependencies(refresh, support)
            continue
        missing = [name for name, module in REQUIRED_MODULES.items() if module_missing(module)]
        if missing:
            install_missing_dependencies(missing, support)
            continue
        failures = import_failures()
        if not failures:
            return
        (OUT / "import_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        missing_from_import = missing_names_from_failures(failures)
        if missing_from_import:
            install_missing_dependencies(missing_from_import, support)
            continue
        raise ImportError("Packages are present but imports fail; see import_failures.json")
    raise RuntimeError("Dependency recovery did not converge")


def copy_vendored_official() -> None:
    here = Path(__file__).resolve().parent
    src = next((p for p in [here / "official", Path("/kaggle/src/official")] if p.exists()), None)
    if src is None:
        raise FileNotFoundError("Vendored official source not found")
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
        "wheel_dirs": [str(x) for x in find_offline_package_dirs(support)],
        "preinstalled": {{name: not module_missing(module) for name, module in REQUIRED_MODULES.items()}},
    }}, indent=2), encoding="utf-8")

    ensure_dependencies(support)
    (OUT / "dependency_after.json").write_text(json.dumps({{
        "available": {{name: not module_missing(module) for name, module in REQUIRED_MODULES.items()}},
        "import_failures": import_failures(),
    }}, indent=2), encoding="utf-8")

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
        return int(sum(int(x) >= 2 for x in graph.out_degree(ids))) if len(ids) else 0

    geffs = sorted(TRAIN.glob("*.geff")) or sorted(TRAIN.rglob("*.geff"))
    if not geffs:
        raise RuntimeError(f"No ground-truth .geff files found below {{TRAIN}}")

    rows, metric_rows, selftest_failures = [], [], []
    for path in geffs:
        name = path.stem
        graph = load_graph(td, path)
        n_nodes, n_edges = int(graph.num_nodes()), int(graph.num_edges())
        n_div, n_total, scale = divisions(graph), estimated_total(path), dataset_scale(name)
        pred, gt = load_graph(td, path), load_graph(td, path)
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
            "scale_z": scale[0], "scale_y": scale[1], "scale_x": scale[2],
            "self_edge_tp": int(er.edge_tp), "self_edge_fp": int(er.edge_fp), "self_edge_fn": int(er.edge_fn),
            "self_div_tp": int(er.division_tp), "self_div_fp": int(er.division_fp), "self_div_fn": int(er.division_fn),
            "self_node_recall": float(rec),
        }})

    with (OUT / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    names = [r["dataset"] for r in rows]
    lodo = [{{"fold": i, "train": [x for x in names if x != holdout], "valid": [holdout]}} for i, holdout in enumerate(names)]
    (OUT / "folds_lodo.json").write_text(json.dumps(lodo, indent=2), encoding="utf-8")

    k = min(5, len(rows))
    folds = [{{"fold": i, "valid": [], "edges": 0, "divisions": 0, "nodes": 0}} for i in range(k)]
    for r in sorted(rows, key=lambda x: (x["gt_edges"], x["gt_divisions"], x["gt_nodes"]), reverse=True):
        target = min(folds, key=lambda f: (f["edges"], f["divisions"], f["nodes"], f["fold"]))
        target["valid"].append(r["dataset"]); target["edges"] += r["gt_edges"]; target["divisions"] += r["gt_divisions"]; target["nodes"] += r["gt_nodes"]
    for fold in folds:
        fold["train"] = [x for x in names if x not in set(fold["valid"])]
    (OUT / "folds_balanced_5.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")

    score_summary = summarise(metric_rows)
    (OUT / "official_metric_selftest.json").write_text(json.dumps({{
        "datasets": len(rows),
        "official_commit": OFFICIAL_COMMIT,
        "summary": {{k: (float(v) if hasattr(v, "__float__") else v) for k, v in score_summary.items()}},
        "failures": selftest_failures,
    }}, indent=2, default=str), encoding="utf-8")

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
        "python": sys.version, "platform": platform.platform(), "official_commit": OFFICIAL_COMMIT,
        "tracksdata": getattr(td, "__version__", "unknown"),
    }}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


try:
    run()
except Exception as exc:
    payload = {{"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
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
    print(json.dumps({"kernel": KERNEL_ID, "official_commit": OFFICIAL_COMMIT, "support_dataset": SUPPORT_DATASET}, indent=2))


if __name__ == "__main__":
    main()
