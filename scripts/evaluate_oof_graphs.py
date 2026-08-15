#!/usr/bin/env python
"""Evaluate OOF Biohub graphs with official score plus detection/linking diagnostics.

This script targets the current support-pack baseline API (`biohub_tracking`).
It evaluates every dataset present in both `--pred-dir` and `--gt-dir`.

Key decomposition:

    edge_endpoint_coverage = recoverable_gt_edges / gt_edges
    conditional_link_recall = edge_tp / recoverable_gt_edges

A GT edge is "recoverable" when both of its GT endpoints are matched to a
predicted node under the official 7 um node matching. This separates losses
caused by missing endpoints (detection/localisation) from losses that remain
when both endpoints are available (association/linking).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

import tracksdata as td
from geff import GeffMetadata

from biohub_tracking.io import DEFAULT_SCALE, open_dataset
from biohub_tracking.metrics import evaluate, node_recall, per_sample_metrics, summarise


def load_graph(path: Path):
    obj = td.graph.IndexedRXGraph.from_geff(path)
    return obj[0] if isinstance(obj, tuple) else obj


def read_scale(gt_dir: Path, name: str) -> tuple[float, float, float]:
    try:
        return tuple(float(x) for x in open_dataset(gt_dir / name, load_image=False).scale)
    except Exception:
        return tuple(float(x) for x in DEFAULT_SCALE)


def read_estimated_total(path: Path) -> float:
    try:
        meta = GeffMetadata.read(path)
        value = (meta.extra or {}).get("estimated_number_of_nodes")
        return float(value) if value is not None else float("nan")
    except Exception:
        return float("nan")


def localization_distances_um(pred_graph, gt_graph, scale: tuple[float, float, float]) -> list[float]:
    """Distances for uniquely matched GT nodes after official matching."""
    k = td.DEFAULT_ATTR_KEYS
    pred = pred_graph.node_attrs(attr_keys=[k.NODE_ID, k.MATCHED_NODE_ID, k.Z, k.Y, k.X])
    gt = gt_graph.node_attrs(attr_keys=[k.NODE_ID, k.Z, k.Y, k.X])
    gt_pos = {
        int(r[k.NODE_ID]): (float(r[k.Z]), float(r[k.Y]), float(r[k.X]))
        for r in gt.to_dicts()
    }
    best: dict[int, float] = {}
    sz, sy, sx = scale
    for r in pred.to_dicts():
        gid = int(r[k.MATCHED_NODE_ID])
        if gid < 0 or gid not in gt_pos:
            continue
        gz, gy, gx = gt_pos[gid]
        dz = (float(r[k.Z]) - gz) * sz
        dy = (float(r[k.Y]) - gy) * sy
        dx = (float(r[k.X]) - gx) * sx
        d = math.sqrt(dz * dz + dy * dy + dx * dx)
        if gid not in best or d < best[gid]:
            best[gid] = d
    return list(best.values())


def matched_gt_node_ids(pred_graph) -> set[int]:
    k = td.DEFAULT_ATTR_KEYS
    attrs = pred_graph.node_attrs(attr_keys=[k.MATCHED_NODE_ID])
    return {
        int(r[k.MATCHED_NODE_ID])
        for r in attrs.to_dicts()
        if int(r[k.MATCHED_NODE_ID]) >= 0
    }


def recoverable_gt_edges(gt_graph, matched_gt: set[int]) -> int:
    k = td.DEFAULT_ATTR_KEYS
    attrs = gt_graph.edge_attrs(attr_keys=[k.EDGE_SOURCE, k.EDGE_TARGET])
    return sum(
        int(r[k.EDGE_SOURCE]) in matched_gt and int(r[k.EDGE_TARGET]) in matched_gt
        for r in attrs.to_dicts()
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-distance", type=float, default=7.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_names = {p.stem for p in args.pred_dir.glob("*.geff")}
    gt_names = {p.stem for p in args.gt_dir.glob("*.geff")}
    names = sorted(pred_names & gt_names)
    if not names:
        raise RuntimeError("No matching .geff dataset names in prediction and GT directories")

    rows: list[dict] = []
    official_rows: list[dict] = []

    for name in names:
        pred = load_graph(args.pred_dir / f"{name}.geff")
        gt = load_graph(args.gt_dir / f"{name}.geff")
        scale = read_scale(args.gt_dir, name)
        n_total = read_estimated_total(args.gt_dir / f"{name}.geff")

        # Official evaluate() performs node matching in-place on pred.
        er = evaluate(pred, gt, scale=scale, max_distance=args.max_distance)
        rec = node_recall(pred, gt) if pred.num_nodes() and gt.num_nodes() else 0.0
        pm = per_sample_metrics(er, n_total, rec)
        official_rows.append(pm)

        matched_gt = matched_gt_node_ids(pred)
        recoverable = recoverable_gt_edges(gt, matched_gt)
        gt_edges = int(gt.num_edges())
        missing_node_edges = max(0, gt_edges - recoverable)
        link_miss_edges = max(0, recoverable - int(er.edge_tp))
        endpoint_cov = recoverable / gt_edges if gt_edges else float("nan")
        cond_link_recall = int(er.edge_tp) / recoverable if recoverable else float("nan")

        dists = localization_distances_um(pred, gt, scale)
        pred_nodes = int(er.num_pred_nodes)
        node_ratio = (
            pred_nodes / n_total
            if n_total > 0 and not math.isnan(n_total)
            else float("nan")
        )

        row = {
            "dataset": name,
            "gt_nodes_sparse": int(gt.num_nodes()),
            "gt_edges": gt_edges,
            "pred_nodes": pred_nodes,
            "estimated_total_nodes": n_total,
            "pred_over_estimated_nodes": node_ratio,
            "node_recall": float(rec),
            "matched_gt_nodes": len(matched_gt),
            "loc_median_um": median(dists) if dists else float("nan"),
            "loc_p90_um": percentile(dists, 0.90),
            "edge_endpoint_coverage": endpoint_cov,
            "recoverable_gt_edges": recoverable,
            "missing_node_gt_edges": missing_node_edges,
            "edge_tp": int(er.edge_tp),
            "edge_fp": int(er.edge_fp),
            "edge_fn": int(er.edge_fn),
            "conditional_link_recall": cond_link_recall,
            "link_miss_gt_edges": link_miss_edges,
            "division_tp": int(er.division_tp),
            "division_fp": int(er.division_fp),
            "division_fn": int(er.division_fn),
            "edge_jaccard": float(pm["edge_jaccard"]),
            "adjusted_edge_jaccard": float(pm["adj_edge_jaccard"]),
        }
        rows.append(row)
        print(json.dumps(row, default=str))

    with (args.out_dir / "cv_per_dataset.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    official_summary = summarise(official_rows)
    total_gt_edges = sum(r["gt_edges"] for r in rows)
    total_recoverable = sum(r["recoverable_gt_edges"] for r in rows)
    total_tp = sum(r["edge_tp"] for r in rows)
    total_missing_node = sum(r["missing_node_gt_edges"] for r in rows)
    total_link_miss = sum(r["link_miss_gt_edges"] for r in rows)

    summary = {
        "n_datasets": len(rows),
        "official": {
            k: (float(v) if hasattr(v, "__float__") else v)
            for k, v in official_summary.items()
        },
        "diagnostics": {
            "edge_endpoint_coverage_micro": (
                total_recoverable / total_gt_edges if total_gt_edges else float("nan")
            ),
            "conditional_link_recall_micro": (
                total_tp / total_recoverable if total_recoverable else float("nan")
            ),
            "gt_edges": total_gt_edges,
            "recoverable_gt_edges": total_recoverable,
            "missing_node_gt_edges": total_missing_node,
            "link_miss_gt_edges": total_link_miss,
            "edge_fp": sum(r["edge_fp"] for r in rows),
        },
    }
    (args.out_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
