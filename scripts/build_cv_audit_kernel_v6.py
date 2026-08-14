from __future__ import annotations

import json
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, KERNEL_ID, KERNEL_TITLE, SUPPORT_DATASET
from build_cv_audit_kernel_v5 import build_worker as build_worker_v5


def build_worker() -> str:
    worker = build_worker_v5()
    start = worker.index("def run() -> None:")
    end = worker.index("\n\ntry:\n    run()", start)
    new_run = r'''def run() -> None:
    import gc

    # Phase 0 is intentionally lightweight.  Full GT->GT distance matching on
    # every training graph can be much more expensive than the dataset audit we
    # actually need, so first establish the dataset structure and safe CV folds.
    support = find_support_root()

    train_exists = TRAIN.exists()
    train_children = []
    if train_exists:
        for p in sorted(TRAIN.iterdir()):
            train_children.append({
                "name": p.name,
                "is_dir": p.is_dir(),
                "suffix": p.suffix,
            })
    geff_paths = sorted(TRAIN.glob("*.geff")) if train_exists else []
    if train_exists and not geff_paths:
        geff_paths = sorted(TRAIN.rglob("*.geff"))

    (OUT / "train_layout.json").write_text(json.dumps({
        "train_root": str(TRAIN),
        "train_exists": train_exists,
        "top_level_children": train_children,
        "geff_paths": [str(p) for p in geff_paths],
    }, indent=2), encoding="utf-8")
    (OUT / "progress.json").write_text(json.dumps({
        "phase": "layout_scanned",
        "n_geffs": len(geff_paths),
    }, indent=2), encoding="utf-8")

    if not train_exists:
        raise FileNotFoundError(f"Competition train root does not exist: {TRAIN}")
    if not geff_paths:
        raise RuntimeError(f"No ground-truth .geff files found below {TRAIN}")

    # Restore the exact dependency stack used by the public learned-graph
    # notebook and materialize its bundled evaluation repository.
    (OUT / "dependency_probe.json").write_text(json.dumps({
        "support_root": str(support),
        "wheel_dirs": [str(x) for x in find_offline_package_dirs(support)],
        "preinstalled": {name: not module_missing(module) for name, module in REQUIRED_MODULES.items()},
    }, indent=2), encoding="utf-8")
    ensure_dependencies(support)
    copy_vendored_official()
    sys.path.insert(0, str(OFFICIAL / "scripts"))
    sys.path.insert(0, str(OFFICIAL / "src"))

    import tracksdata as td
    from geff import GeffMetadata
    from biohub_tracking.io import DEFAULT_SCALE, open_dataset
    import biohub_tracking.metrics as metric_module

    (OUT / "dependency_after.json").write_text(json.dumps({
        "available": {name: not module_missing(module) for name, module in REQUIRED_MODULES.items()},
        "import_failures": import_failures(),
        "tracksdata_version": getattr(td, "__version__", "unknown"),
        "metric_module": str(Path(metric_module.__file__).resolve()),
    }, indent=2), encoding="utf-8")

    def estimated_total(path: Path):
        try:
            meta = GeffMetadata.read(path)
            value = (meta.extra or {}).get("estimated_number_of_nodes")
            return float(value) if value is not None else float("nan")
        except Exception:
            return float("nan")

    def dataset_scale(name: str):
        try:
            return tuple(float(x) for x in open_dataset(TRAIN / name, load_image=False).scale)
        except Exception:
            return tuple(float(x) for x in DEFAULT_SCALE)

    def count_divisions(graph) -> int:
        ids = graph.node_ids()
        if len(ids) == 0:
            return 0
        degrees = graph.out_degree(ids)
        return int(sum(int(x) >= 2 for x in degrees))

    def write_manifest(rows):
        if not rows:
            return
        with (OUT / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with (OUT / "dataset_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=str)

    rows = []
    failures = []
    for index, path in enumerate(geff_paths):
        (OUT / "progress.json").write_text(json.dumps({
            "phase": "loading_graph",
            "index": index,
            "n_geffs": len(geff_paths),
            "path": str(path),
            "completed": [r["dataset"] for r in rows],
        }, indent=2), encoding="utf-8")
        try:
            graph = load_graph(td, path)
            name = path.stem
            node_ids = graph.node_ids()
            n_nodes = int(graph.num_nodes())
            n_edges = int(graph.num_edges())
            n_divisions = count_divisions(graph)
            scale = dataset_scale(name)
            row = {
                "dataset": name,
                "gt_nodes": n_nodes,
                "gt_edges": n_edges,
                "gt_divisions": n_divisions,
                "estimated_number_of_nodes": estimated_total(path),
                "scale_z": scale[0],
                "scale_y": scale[1],
                "scale_x": scale[2],
                "geff_path": str(path),
            }
            rows.append(row)
            del node_ids, graph
            gc.collect()
            write_manifest(rows)
            (OUT / "progress.json").write_text(json.dumps({
                "phase": "graph_completed",
                "index": index,
                "n_geffs": len(geff_paths),
                "dataset": name,
                "completed": [r["dataset"] for r in rows],
            }, indent=2), encoding="utf-8")
        except Exception as exc:
            failures.append({
                "path": str(path),
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })
            (OUT / "graph_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
            gc.collect()

    if not rows:
        raise RuntimeError("No ground-truth graph could be loaded; see graph_failures.json")

    names = [r["dataset"] for r in rows]
    lodo = [
        {"fold": i, "train": [x for x in names if x != holdout], "valid": [holdout]}
        for i, holdout in enumerate(names)
    ]
    (OUT / "folds_lodo.json").write_text(json.dumps(lodo, indent=2), encoding="utf-8")

    k = min(5, len(rows))
    folds = [{"fold": i, "valid": [], "edges": 0, "divisions": 0, "nodes": 0} for i in range(k)]
    for row in sorted(rows, key=lambda x: (x["gt_edges"], x["gt_divisions"], x["gt_nodes"]), reverse=True):
        target = min(folds, key=lambda f: (f["edges"], f["divisions"], f["nodes"], f["fold"]))
        target["valid"].append(row["dataset"])
        target["edges"] += row["gt_edges"]
        target["divisions"] += row["gt_divisions"]
        target["nodes"] += row["gt_nodes"]
    for fold in folds:
        fold["train"] = [x for x in names if x not in set(fold["valid"])]
    (OUT / "folds_balanced_5.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")

    # Full metric matching is deliberately deferred until the manifest is known.
    # Importing the evaluator here proves that the scoring implementation and its
    # dependency stack are available without risking an all-dataset GT self-match.
    metric_selftest = {
        "status": "import_only_phase0",
        "metric_module": str(Path(metric_module.__file__).resolve()),
        "evaluate_available": callable(getattr(metric_module, "evaluate", None)),
        "summarise_available": callable(getattr(metric_module, "summarise", None)),
        "reason_full_gt_selfmatch_deferred": "Avoid unnecessary peak memory during dataset/fold audit; run targeted metric smoke test after selecting the smallest dataset.",
    }
    (OUT / "official_metric_selftest.json").write_text(json.dumps(metric_selftest, indent=2), encoding="utf-8")

    edge_counts = [r["gt_edges"] for r in rows]
    node_counts = [r["gt_nodes"] for r in rows]
    division_counts = [r["gt_divisions"] for r in rows]
    summary = {
        "competition": "biohub-cell-tracking-during-development",
        "train_root": str(TRAIN),
        "support_root": str(support),
        "n_geffs_found": len(geff_paths),
        "n_datasets_loaded": len(rows),
        "n_graph_failures": len(failures),
        "total_gt_nodes": sum(node_counts),
        "total_gt_edges": sum(edge_counts),
        "total_gt_divisions": sum(division_counts),
        "min_gt_nodes": min(node_counts),
        "max_gt_nodes": max(node_counts),
        "min_gt_edges": min(edge_counts),
        "max_gt_edges": max(edge_counts),
        "min_gt_divisions": min(division_counts),
        "max_gt_divisions": max(division_counts),
        "primary_cv_candidate": "leave-one-dataset-out",
        "secondary_cv_candidate": f"balanced_{k}_fold",
        "metric_selftest": "import_only_phase0",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "audit_environment.json").write_text(json.dumps({
        "python": sys.version,
        "platform": platform.platform(),
        "tracksdata": getattr(td, "__version__", "unknown"),
        "metric_module": str(Path(metric_module.__file__).resolve()),
    }, indent=2), encoding="utf-8")
    (OUT / "progress.json").write_text(json.dumps({
        "phase": "complete",
        "datasets": names,
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
'''
    return worker[:start] + new_run + worker[end:]


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
        "mode": "manifest-first-v6",
    }, indent=2))


if __name__ == "__main__":
    main()
