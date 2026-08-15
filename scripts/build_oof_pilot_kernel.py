from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET, vendor_official_source
from build_cv_audit_kernel_v7 import build_worker as build_audit_worker_v7

KERNEL_ID = "ykhnkf/biohub-cell-tracking-oof-pilot"
KERNEL_TITLE = "Biohub Cell Tracking OOF Pilot"


def build_worker() -> str:
    base = build_audit_worker_v7()
    start = base.index("def run() -> None:")
    prefix = base[:start]
    prefix = prefix.replace(
        'OUT = Path("/kaggle/working/biohub_cv_audit")',
        'OUT = Path("/kaggle/working/biohub_oof_pilot")',
        1,
    )

    run = r'''def run() -> None:
    import gc
    import statistics
    import time

    support = find_support_root()
    (OUT / "progress.json").write_text(json.dumps({"phase": "dependency_setup"}, indent=2), encoding="utf-8")
    ensure_dependencies(support)
    copy_vendored_official()
    sys.path.insert(0, str(OFFICIAL / "scripts"))
    sys.path.insert(0, str(OFFICIAL / "src"))

    import torch
    import tracksdata as td
    from geff import GeffMetadata
    from biohub_tracking.io import DEFAULT_SCALE, open_dataset, save_graph
    from biohub_tracking.metrics import evaluate, node_recall, per_sample_metrics, summarise
    from predict_unet_transformer import PredictConfig, build_graph, load_model, predict_video

    if not torch.cuda.is_available():
        raise RuntimeError("OOF pilot requires a Kaggle GPU, but torch.cuda.is_available() is False")
    device = torch.device("cuda")

    def normalize_name(value) -> str:
        name = Path(str(value)).name
        if name.endswith(".zarr") or name.endswith(".geff"):
            name = name.rsplit(".", 1)[0]
        return name

    def parse_split_file(path: Path):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if isinstance(obj, dict) and "folds" in obj:
            obj = obj["folds"]
        folds = {}
        if isinstance(obj, list):
            items = list(enumerate(obj))
        elif isinstance(obj, dict):
            items = []
            for key, value in obj.items():
                m = re.search(r"(\d+)", str(key))
                if m and isinstance(value, dict):
                    items.append((int(m.group(1)), value))
        else:
            items = []
        for idx, fold in items:
            if not isinstance(fold, dict):
                continue
            vals = None
            for key in ("test", "valid", "val", "validation", "holdout"):
                if key in fold and isinstance(fold[key], list):
                    vals = fold[key]
                    break
            if vals:
                folds[int(idx)] = [normalize_name(x) for x in vals]
        return folds

    weight_candidates = sorted(
        [p for ext in ("*.pth", "*.pt", "*.ckpt") for p in support.rglob(ext)]
    )
    split_candidates = sorted(
        set(support.rglob("dataset_splits.json"))
        | set(support.rglob("splits.json"))
        | set(support.rglob("*fold*.json"))
    )

    weights_by_fold = {}
    weight_rank = {}
    for path in weight_candidates:
        text = str(path)
        m = re.search(r"(?:split|fold)[_-]?(\d+)", text, flags=re.IGNORECASE)
        if not m:
            continue
        fold = int(m.group(1))
        low = path.name.lower()
        rank = (
            0 if "edge_predictor_best" in low else
            1 if "edge_predictor" in low and "best" in low else
            2 if "best" in low else
            3
        )
        if fold not in weights_by_fold or rank < weight_rank[fold]:
            weights_by_fold[fold] = path
            weight_rank[fold] = rank

    parsed_splits = []
    for path in split_candidates:
        folds = parse_split_file(path)
        if folds:
            parsed_splits.append((path, folds))
    if not parsed_splits:
        raise RuntimeError("Could not discover a usable fold split JSON in the support pack")
    split_path, folds = max(parsed_splits, key=lambda item: (len(item[1]), sum(len(v) for v in item[1].values())))

    common_folds = sorted(set(weights_by_fold) & set(folds))
    inventory = {
        "support_root": str(support),
        "gpu": torch.cuda.get_device_name(0),
        "all_weight_candidates": [str(p) for p in weight_candidates],
        "selected_weights_by_fold": {str(k): str(v) for k, v in weights_by_fold.items()},
        "split_candidates": [str(p) for p in split_candidates],
        "selected_split_file": str(split_path),
        "split_fold_sizes": {str(k): len(v) for k, v in folds.items()},
        "common_folds": common_folds,
    }
    (OUT / "support_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    if not common_folds:
        raise RuntimeError("No fold IDs are shared by discovered weights and split definitions")

    def load_graph(path: Path):
        obj = td.graph.IndexedRXGraph.from_geff(path)
        return obj[0] if isinstance(obj, tuple) else obj

    def count_divisions(graph) -> int:
        ids = graph.node_ids()
        if len(ids) == 0:
            return 0
        return int(sum(int(x) >= 2 for x in graph.out_degree(ids)))

    def gt_stats(name: str):
        geff = TRAIN / f"{name}.geff"
        zarr_path = TRAIN / f"{name}.zarr"
        if not geff.exists() or not zarr_path.exists():
            return None
        graph = load_graph(geff)
        row = {
            "dataset": name,
            "gt_nodes": int(graph.num_nodes()),
            "gt_edges": int(graph.num_edges()),
            "gt_divisions": count_divisions(graph),
        }
        del graph
        return row

    pilot_plan = []
    for fold in common_folds:
        stats = []
        for name in folds[fold]:
            row = gt_stats(name)
            if row is not None:
                stats.append(row)
        if not stats:
            continue
        edge_values = sorted(r["gt_edges"] for r in stats)
        med = statistics.median(edge_values)
        median_pick = min(stats, key=lambda r: (abs(r["gt_edges"] - med), r["gt_nodes"], r["dataset"]))
        selected = [median_pick]
        divs = [r for r in stats if r["gt_divisions"] > 0 and r["dataset"] != median_pick["dataset"]]
        if divs:
            selected.append(min(divs, key=lambda r: (r["gt_nodes"], r["gt_edges"], r["dataset"])))
        elif len(stats) > 1:
            alt = max((r for r in stats if r["dataset"] != median_pick["dataset"]), key=lambda r: (r["gt_edges"], r["gt_nodes"]))
            selected.append(alt)
        for row in selected[:2]:
            pilot_plan.append({
                **row,
                "fold": int(fold),
                "weight": str(weights_by_fold[fold]),
            })

    if not pilot_plan:
        raise RuntimeError("No valid OOF pilot datasets could be selected")
    (OUT / "pilot_plan.json").write_text(json.dumps(pilot_plan, indent=2), encoding="utf-8")

    pred_dir = OUT / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    cfg = PredictConfig(
        det_threshold=0.99,
        det_tta=True,
        pool_kernel_um=3.0,
        edge_activation="softmax",
        threshold=0.5,
        use_ilp=False,
    )

    for fold in common_folds:
        fold_rows = [r for r in pilot_plan if r["fold"] == fold]
        if not fold_rows:
            continue
        weight_path = weights_by_fold[fold]
        (OUT / "progress.json").write_text(json.dumps({
            "phase": "loading_model", "fold": fold, "weight": str(weight_path)
        }, indent=2), encoding="utf-8")
        t0 = time.time()
        model, window_size, downsample = load_model(weight_path, device)
        model_load_sec = time.time() - t0

        for row in fold_rows:
            name = row["dataset"]
            zarr_path = TRAIN / f"{name}.zarr"
            (OUT / "progress.json").write_text(json.dumps({
                "phase": "predicting", "fold": fold, "dataset": name,
                "completed": [r["dataset"] for r in prediction_rows],
            }, indent=2), encoding="utf-8")
            t1 = time.time()
            coords, edges = predict_video(
                model,
                zarr_path,
                device,
                cfg=cfg,
                window_size=window_size,
                downsample=downsample,
            )
            graph = build_graph(coords, edges)
            save_graph(graph, pred_dir / f"{name}.geff")
            elapsed = time.time() - t1
            prediction_rows.append({
                "dataset": name,
                "fold": fold,
                "pred_nodes": int(graph.num_nodes()),
                "pred_edges": int(graph.num_edges()),
                "seconds": elapsed,
                "model_load_seconds": model_load_sec,
                "window_size": int(window_size),
                "downsample": list(downsample),
                "det_threshold": cfg.det_threshold,
                "det_tta": cfg.det_tta,
                "edge_threshold": cfg.threshold,
                "weight": str(weight_path),
            })
            del coords, edges, graph
            gc.collect()
            torch.cuda.empty_cache()

        del model
        gc.collect()
        torch.cuda.empty_cache()

    with (OUT / "prediction_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)

    def estimated_total(path: Path) -> float:
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

    def matched_gt_node_ids(pred_graph) -> set[int]:
        k = td.DEFAULT_ATTR_KEYS
        attrs = pred_graph.node_attrs(attr_keys=[k.MATCHED_NODE_ID])
        out = set()
        for r in attrs.to_dicts():
            gid = int(r[k.MATCHED_NODE_ID])
            if gid >= 0:
                out.add(gid)
        return out

    def recoverable_gt_edges(gt_graph, matched_gt: set[int]) -> int:
        k = td.DEFAULT_ATTR_KEYS
        attrs = gt_graph.edge_attrs(attr_keys=[k.EDGE_SOURCE, k.EDGE_TARGET])
        return sum(
            int(r[k.EDGE_SOURCE]) in matched_gt and int(r[k.EDGE_TARGET]) in matched_gt
            for r in attrs.to_dicts()
        )

    metric_rows = []
    official_rows = []
    for pred_info in prediction_rows:
        name = pred_info["dataset"]
        pred = load_graph(pred_dir / f"{name}.geff")
        gt = load_graph(TRAIN / f"{name}.geff")
        scale = dataset_scale(name)
        total = estimated_total(TRAIN / f"{name}.geff")
        er = evaluate(pred, gt, scale=scale, max_distance=7.0)
        rec = node_recall(pred, gt) if pred.num_nodes() and gt.num_nodes() else 0.0
        pm = per_sample_metrics(er, total, rec)
        official_rows.append(pm)
        matched = matched_gt_node_ids(pred)
        recoverable = recoverable_gt_edges(gt, matched)
        gt_edges = int(gt.num_edges())
        row = {
            "dataset": name,
            "fold": int(pred_info["fold"]),
            "gt_nodes_sparse": int(gt.num_nodes()),
            "gt_edges": gt_edges,
            "pred_nodes": int(er.num_pred_nodes),
            "estimated_total_nodes": total,
            "pred_over_estimated_nodes": float(er.num_pred_nodes) / total if total else float("nan"),
            "node_recall": float(rec),
            "matched_gt_nodes": len(matched),
            "recoverable_gt_edges": int(recoverable),
            "edge_endpoint_coverage": recoverable / gt_edges if gt_edges else float("nan"),
            "edge_tp": int(er.edge_tp),
            "edge_fp": int(er.edge_fp),
            "edge_fn": int(er.edge_fn),
            "conditional_link_recall": int(er.edge_tp) / recoverable if recoverable else float("nan"),
            "missing_node_gt_edges": max(0, gt_edges - recoverable),
            "link_miss_gt_edges": max(0, recoverable - int(er.edge_tp)),
            "division_tp": int(er.division_tp),
            "division_fp": int(er.division_fp),
            "division_fn": int(er.division_fn),
            "edge_jaccard": float(pm.get("edge_jaccard", float("nan"))),
            "adjusted_edge_jaccard": float(pm.get("adj_edge_jaccard", float("nan"))),
            "division_jaccard": float(pm.get("division_jaccard", float("nan"))),
            "score": float(pm.get("score", float("nan"))),
        }
        metric_rows.append(row)
        del pred, gt
        gc.collect()

    with (OUT / "cv_per_dataset.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    official_summary = summarise(official_rows)
    total_gt_edges = sum(r["gt_edges"] for r in metric_rows)
    total_recoverable = sum(r["recoverable_gt_edges"] for r in metric_rows)
    total_tp = sum(r["edge_tp"] for r in metric_rows)
    summary = {
        "status": "passed",
        "n_datasets": len(metric_rows),
        "n_folds": len(set(r["fold"] for r in metric_rows)),
        "official": {
            k: (int(v) if isinstance(v, int) else float(v))
            for k, v in official_summary.items()
        },
        "diagnostics": {
            "gt_edges": total_gt_edges,
            "recoverable_gt_edges": total_recoverable,
            "edge_endpoint_coverage_micro": total_recoverable / total_gt_edges if total_gt_edges else float("nan"),
            "conditional_link_recall_micro": total_tp / total_recoverable if total_recoverable else float("nan"),
            "missing_node_gt_edges": sum(r["missing_node_gt_edges"] for r in metric_rows),
            "link_miss_gt_edges": sum(r["link_miss_gt_edges"] for r in metric_rows),
            "edge_fp": sum(r["edge_fp"] for r in metric_rows),
        },
        "config": {
            "det_threshold": cfg.det_threshold,
            "det_tta": cfg.det_tta,
            "pool_kernel_um": cfg.pool_kernel_um,
            "edge_activation": cfg.edge_activation,
            "edge_threshold": cfg.threshold,
            "use_ilp": cfg.use_ilp,
        },
        "split_file": str(split_path),
        "weights_by_fold": {str(k): str(weights_by_fold[k]) for k in common_folds},
    }
    (OUT / "cv_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (OUT / "progress.json").write_text(json.dumps({
        "phase": "complete", "datasets": [r["dataset"] for r in metric_rows]
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


try:
    run()
except Exception as exc:
    payload = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    (OUT / "fatal_error.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise
'''
    return prefix + run


def main() -> None:
    out = Path(".kaggle_oof_pilot_kernel")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    vendor_official_source(out)
    (out / "run_oof_pilot.py").write_text(build_worker(), encoding="utf-8")
    metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "run_oof_pilot.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
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
        "mode": "true-oof-pilot-fold-weights",
        "github_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
    }, indent=2))


if __name__ == "__main__":
    main()
