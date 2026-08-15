from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_cv_audit_kernel_v7 import build_worker as build_audit_worker_v7

KERNEL_ID = "ykhnkf/biohub-pretrained-reference"
KERNEL_TITLE = "Biohub Pretrained Reference Diagnostics"


def build_worker() -> str:
    base = build_audit_worker_v7()
    start = base.index("def run() -> None:")
    prefix = base[:start].replace(
        'OUT = Path("/kaggle/working/biohub_cv_audit")',
        'OUT = Path("/kaggle/working/biohub_pretrained_reference")', 1,
    )
    run = r'''def run() -> None:
    import gc
    import time
    support = find_support_root()
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
        raise RuntimeError("GPU unavailable")
    device = torch.device("cuda")
    weights = support / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
    if not weights.exists():
        raise FileNotFoundError(weights)

    datasets = [
        "44b6_341df25f",  # representative 44b6, medium sparse-GT size
        "44b6_d754aa59",  # small 44b6 division case
        "6bba_6321a359",  # representative 6bba, medium sparse-GT size
        "6bba_0e7c0d07",  # small 6bba division case
    ]
    missing = [n for n in datasets if not (TRAIN / f"{n}.zarr").exists() or not (TRAIN / f"{n}.geff").exists()]
    if missing:
        raise FileNotFoundError(f"Missing pilot datasets: {missing}")

    cfg = PredictConfig(
        det_threshold=0.99,
        det_tta=True,
        pool_kernel_um=3.0,
        edge_activation="softmax",
        threshold=0.5,
        use_ilp=False,
    )
    model, window_size, downsample = load_model(weights, device)
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_manifest = []

    for name in datasets:
        (OUT / "progress.json").write_text(json.dumps({"phase":"predicting","dataset":name,"completed":[x["dataset"] for x in pred_manifest]}, indent=2), encoding="utf-8")
        t0 = time.time()
        coords, edges = predict_video(model, TRAIN / f"{name}.zarr", device, cfg=cfg, window_size=window_size, downsample=downsample)
        graph = build_graph(coords, edges)
        save_graph(graph, pred_dir / f"{name}.geff")
        pred_manifest.append({
            "dataset": name,
            "pred_nodes": int(graph.num_nodes()),
            "pred_edges": int(graph.num_edges()),
            "seconds": time.time() - t0,
        })
        del coords, edges, graph
        gc.collect(); torch.cuda.empty_cache()

    with (OUT / "prediction_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_manifest[0]))
        w.writeheader(); w.writerows(pred_manifest)

    def load_graph(path: Path):
        obj = td.graph.IndexedRXGraph.from_geff(path)
        return obj[0] if isinstance(obj, tuple) else obj

    def estimated_total(path: Path) -> float:
        meta = GeffMetadata.read(path)
        value = (meta.extra or {}).get("estimated_number_of_nodes")
        return float(value) if value is not None else float("nan")

    def dataset_scale(name: str):
        try:
            return tuple(float(x) for x in open_dataset(TRAIN / name, load_image=False).scale)
        except Exception:
            return tuple(float(x) for x in DEFAULT_SCALE)

    def matched_gt_node_ids(pred_graph):
        k = td.DEFAULT_ATTR_KEYS
        attrs = pred_graph.node_attrs(attr_keys=[k.MATCHED_NODE_ID])
        return {int(r[k.MATCHED_NODE_ID]) for r in attrs.to_dicts() if int(r[k.MATCHED_NODE_ID]) >= 0}

    def recoverable_gt_edges(gt_graph, matched_gt):
        k = td.DEFAULT_ATTR_KEYS
        attrs = gt_graph.edge_attrs(attr_keys=[k.EDGE_SOURCE, k.EDGE_TARGET])
        return sum(int(r[k.EDGE_SOURCE]) in matched_gt and int(r[k.EDGE_TARGET]) in matched_gt for r in attrs.to_dicts())

    rows, official_rows = [], []
    for info in pred_manifest:
        name = info["dataset"]
        pred = load_graph(pred_dir / f"{name}.geff")
        gt = load_graph(TRAIN / f"{name}.geff")
        er = evaluate(pred, gt, scale=dataset_scale(name), max_distance=7.0)
        rec = node_recall(pred, gt)
        total = estimated_total(TRAIN / f"{name}.geff")
        pm = per_sample_metrics(er, total, rec)
        official_rows.append(pm)
        matched = matched_gt_node_ids(pred)
        recoverable = recoverable_gt_edges(gt, matched)
        gt_edges = int(gt.num_edges())
        rows.append({
            "dataset": name,
            "gt_nodes_sparse": int(gt.num_nodes()),
            "gt_edges": gt_edges,
            "pred_nodes": int(er.num_pred_nodes),
            "estimated_total_nodes": total,
            "pred_over_estimated_nodes": float(er.num_pred_nodes) / total if total else float("nan"),
            "node_recall": float(rec),
            "recoverable_gt_edges": int(recoverable),
            "edge_endpoint_coverage": recoverable / gt_edges if gt_edges else float("nan"),
            "edge_tp": int(er.edge_tp), "edge_fp": int(er.edge_fp), "edge_fn": int(er.edge_fn),
            "conditional_link_recall": int(er.edge_tp) / recoverable if recoverable else float("nan"),
            "missing_node_gt_edges": max(0, gt_edges - recoverable),
            "link_miss_gt_edges": max(0, recoverable - int(er.edge_tp)),
            "division_tp": int(er.division_tp), "division_fp": int(er.division_fp), "division_fn": int(er.division_fn),
            "edge_jaccard": float(pm["edge_jaccard"]),
            "adjusted_edge_jaccard": float(pm["adjusted_edge_jaccard"]),
            "division_jaccard": float(pm["division_jaccard"]),
            "score": float(pm["score"]),
        })
        del pred, gt
        gc.collect()

    with (OUT / "reference_per_dataset.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    summary = summarise(official_rows)
    gt_edges = sum(r["gt_edges"] for r in rows)
    recoverable = sum(r["recoverable_gt_edges"] for r in rows)
    tp = sum(r["edge_tp"] for r in rows)
    payload = {
        "status": "passed",
        "provenance": "non_oof_pretrained_reference_unknown_train_overlap",
        "datasets": datasets,
        "weight": str(weights),
        "official": {k: (int(v) if isinstance(v, int) else float(v)) for k,v in summary.items()},
        "diagnostics": {
            "edge_endpoint_coverage_micro": recoverable / gt_edges if gt_edges else float("nan"),
            "conditional_link_recall_micro": tp / recoverable if recoverable else float("nan"),
            "missing_node_gt_edges": sum(r["missing_node_gt_edges"] for r in rows),
            "link_miss_gt_edges": sum(r["link_miss_gt_edges"] for r in rows),
            "edge_fp": sum(r["edge_fp"] for r in rows),
        },
        "config": {"det_threshold":0.99,"det_tta":True,"pool_kernel_um":3.0,"edge_threshold":0.5,"use_ilp":False},
    }
    (OUT / "reference_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (OUT / "progress.json").write_text(json.dumps({"phase":"complete"}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))

try:
    run()
except Exception as exc:
    payload={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
    (OUT/"fatal_error.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2)); raise
'''
    return prefix + run


def main() -> None:
    out=Path(".kaggle_pretrained_reference_kernel")
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    (out/"run_reference.py").write_text(build_worker(), encoding="utf-8")
    meta={
        "id":KERNEL_ID,"title":KERNEL_TITLE,"code_file":"run_reference.py","language":"python","kernel_type":"script",
        "is_private":True,"enable_gpu":True,"enable_tpu":False,"enable_internet":False,"keywords":[],
        "dataset_sources":[SUPPORT_DATASET],"kernel_sources":[],"competition_sources":[COMPETITION],"model_sources":[]
    }
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps({"kernel":KERNEL_ID,"github_run_id":os.environ.get("GITHUB_RUN_ID","local")},indent=2))

if __name__=="__main__": main()
