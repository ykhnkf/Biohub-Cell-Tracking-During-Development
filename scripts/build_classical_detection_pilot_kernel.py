from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_cv_audit_kernel_v7 import build_worker as build_audit_worker_v7

KERNEL_ID = "ykhnkf/biohub-classical-detection-pilot"
KERNEL_TITLE = "Biohub Classical Detection Pilot"


def build_worker() -> str:
    base = build_audit_worker_v7()
    start = base.index("def run() -> None:")
    prefix = base[:start].replace(
        'OUT = Path("/kaggle/working/biohub_cv_audit")',
        'OUT = Path("/kaggle/working/biohub_classical_detection_pilot")', 1,
    )
    run = r'''def run() -> None:
    import gc
    import math
    import time
    import numpy as np
    import zarr
    from scipy.ndimage import gaussian_filter, maximum_filter
    from scipy.optimize import linear_sum_assignment

    support = find_support_root()
    ensure_dependencies(support)
    copy_vendored_official()
    sys.path.insert(0, str(OFFICIAL / "scripts"))
    sys.path.insert(0, str(OFFICIAL / "src"))

    import tracksdata as td
    from geff import GeffMetadata
    from biohub_tracking.io import DEFAULT_SCALE, open_dataset
    from biohub_tracking.metrics import evaluate, node_recall, per_sample_metrics, summarise

    # No learned weights are used.  Calibration and confirmation sets are
    # disjoint; only the calibration group chooses the threshold.
    groups = {
        "calibration": [
            "44b6_341df25f", "44b6_d754aa59",
            "6bba_6321a359", "6bba_0e7c0d07",
        ],
        "confirmation": [
            "44b6_81c256f0", "44b6_9be80b04",
            "6bba_474be664", "6bba_05b6850b",
        ],
    }
    q_values = [0.90, 0.95, 0.97, 0.98, 0.99, 0.995]
    q_labels = {q: f"q{q:.3f}" for q in q_values}
    downsample = np.asarray([1, 4, 4], dtype=np.int64)
    link_gate_um = 10.0
    smooth_sigma = (0.7, 0.8, 0.8)
    nms_size = (3, 3, 3)
    max_peaks_per_frame = 1200

    all_names = sum(groups.values(), [])
    if len(all_names) != len(set(all_names)):
        raise RuntimeError("Calibration and confirmation datasets overlap")
    missing = [n for n in all_names if not (TRAIN / f"{n}.zarr").exists() or not (TRAIN / f"{n}.geff").exists()]
    if missing:
        raise FileNotFoundError(f"Missing pilot datasets: {missing}")

    (OUT / "pilot_design.json").write_text(json.dumps({
        "provenance": "classical_detection_calibration4_confirmation4_no_learned_weights",
        "groups": groups,
        "q_values": q_values,
        "downsample": downsample.tolist(),
        "link_gate_um": link_gate_um,
        "smooth_sigma_downsampled_grid": smooth_sigma,
        "nms_size_downsampled_grid": nms_size,
        "max_peaks_per_frame": max_peaks_per_frame,
        "selection_rule": "choose q by calibration official score only; report confirmation at selected q",
    }, indent=2), encoding="utf-8")

    def load_graph(path: Path):
        obj = td.graph.IndexedRXGraph.from_geff(path)
        return obj[0] if isinstance(obj, tuple) else obj

    def estimated_total(path: Path) -> float:
        try:
            meta = GeffMetadata.read(path)
            value = (meta.extra or {}).get("estimated_number_of_nodes")
            return float(value) if value is not None else float("nan")
        except Exception:
            return float("nan")

    def clear_graph_keep_schema(gt_path: Path):
        g = load_graph(gt_path)
        edge_ids = list(g.edge_ids())
        if edge_ids:
            g.bulk_remove_edges(edge_ids)
        node_ids = list(g.node_ids())
        if node_ids:
            g.bulk_remove_nodes(node_ids)
        return g

    def make_pred_graph(gt_path: Path, coords: np.ndarray, scale: np.ndarray):
        g = clear_graph_keep_schema(gt_path)
        if len(coords) == 0:
            return g
        nodes = [
            {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
            for t, z, y, x in coords
        ]
        new_ids = g.bulk_add_nodes(nodes)
        by_t = {}
        for i, row in enumerate(coords):
            by_t.setdefault(int(row[0]), []).append(i)
        edge_rows = []
        for t in sorted(by_t):
            if t + 1 not in by_t:
                continue
            ia, ib = by_t[t], by_t[t + 1]
            A = coords[ia, 1:4].astype(np.float64)
            B = coords[ib, 1:4].astype(np.float64)
            if len(A) == 0 or len(B) == 0:
                continue
            D = np.linalg.norm((A[:, None, :] - B[None, :, :]) * scale[None, None, :], axis=2)
            cost = D.copy()
            bad = D > link_gate_um
            cost[bad] = link_gate_um + 1e6 + D[bad]
            rr, cc = linear_sum_assignment(cost)
            for rix, cix in zip(rr, cc):
                if D[rix, cix] <= link_gate_um:
                    edge_rows.append({
                        "source_id": int(new_ids[ia[rix]]),
                        "target_id": int(new_ids[ib[cix]]),
                    })
        if edge_rows:
            g.bulk_add_edges(edge_rows)
        return g

    def matched_gt_ids(pred_graph):
        k = td.DEFAULT_ATTR_KEYS
        attrs = pred_graph.node_attrs(attr_keys=[k.MATCHED_NODE_ID])
        return {
            int(r[k.MATCHED_NODE_ID]) for r in attrs.to_dicts()
            if int(r[k.MATCHED_NODE_ID]) >= 0
        }

    def recoverable_gt_edges(gt_graph, matched_gt):
        k = td.DEFAULT_ATTR_KEYS
        attrs = gt_graph.edge_attrs(attr_keys=[k.EDGE_SOURCE, k.EDGE_TARGET])
        return sum(
            int(r[k.EDGE_SOURCE]) in matched_gt and int(r[k.EDGE_TARGET]) in matched_gt
            for r in attrs.to_dicts()
        )

    def detect_dataset(name: str):
        ds = open_dataset(TRAIN / name, normalize=False, load_image=False)
        scale = np.asarray(ds.scale, dtype=np.float64)
        arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
        if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
            raise ValueError(f"Missing 0.001/0.999 image quantiles for {name}")
        qlow = float(ds.quantiles["0.001"])
        qhigh = float(ds.quantiles["0.999"])
        T = int(ds.image_shape[0])
        coords_by_q = {q: [] for q in q_values}
        frame_rows = []
        cap_hits = 0
        t0 = time.time()
        for t in range(T):
            raw = arr[t, ::downsample[0], ::downsample[1], ::downsample[2]].astype(np.float32)
            norm = (raw - qlow) / (qhigh - qlow + 1e-6)
            norm = np.clip(norm, 0.0, 1.5)
            smooth = gaussian_filter(norm, sigma=smooth_sigma, mode="nearest")
            local = maximum_filter(smooth, size=nms_size, mode="nearest")
            peak_mask = smooth == local
            peak_idx = np.argwhere(peak_mask)
            peak_scores = smooth[peak_mask]
            thresholds = {q: float(np.quantile(smooth, q)) for q in q_values}
            counts = {}
            for q in q_values:
                keep = np.flatnonzero(peak_scores >= thresholds[q])
                if len(keep) > max_peaks_per_frame:
                    cap_hits += 1
                    order = np.argpartition(peak_scores[keep], -max_peaks_per_frame)[-max_peaks_per_frame:]
                    keep = keep[order]
                pts = peak_idx[keep].astype(np.float64)
                if len(pts):
                    pts *= downsample[None, :]
                    tcol = np.full((len(pts), 1), t, dtype=np.float64)
                    coords_by_q[q].append(np.concatenate([tcol, pts], axis=1))
                counts[q_labels[q]] = int(len(pts))
            frame_rows.append({
                "dataset": name,
                "t": t,
                "candidate_local_maxima": int(len(peak_idx)),
                **counts,
            })
            del raw, norm, smooth, local, peak_mask, peak_idx, peak_scores
            gc.collect()
            if (t + 1) % 10 == 0 or t + 1 == T:
                (OUT / "progress.json").write_text(json.dumps({
                    "phase": "detecting", "dataset": name, "frame": t + 1, "frames": T,
                }, indent=2), encoding="utf-8")
        stacked = {
            q: (np.concatenate(parts, axis=0) if parts else np.empty((0, 4), dtype=np.float64))
            for q, parts in coords_by_q.items()
        }
        return stacked, frame_rows, scale, T, time.time() - t0, cap_hits

    all_metric_rows = []
    all_frame_rows = []
    detection_manifest = []
    official_by_group_q = {(group, q): [] for group in groups for q in q_values}

    for group, names in groups.items():
        for name in names:
            coords_by_q, frame_rows, scale, T, detect_sec, cap_hits = detect_dataset(name)
            all_frame_rows.extend(frame_rows)
            total_est = estimated_total(TRAIN / f"{name}.geff")
            gt = load_graph(TRAIN / f"{name}.geff")
            gt_edges = int(gt.num_edges())
            detection_manifest.append({
                "group": group, "dataset": name, "frames": T,
                "detect_seconds": detect_sec, "cap_hits": cap_hits,
                "estimated_total_nodes": total_est,
            })
            for q in q_values:
                coords = coords_by_q[q]
                pred = make_pred_graph(TRAIN / f"{name}.geff", coords, scale)
                er = evaluate(pred, gt, scale=tuple(scale), max_distance=7.0)
                rec = node_recall(pred, gt) if pred.num_nodes() and gt.num_nodes() else 0.0
                pm = per_sample_metrics(er, total_est, rec)
                official_by_group_q[(group, q)].append(pm)
                matched = matched_gt_ids(pred) if pred.num_nodes() else set()
                recoverable = recoverable_gt_edges(gt, matched)
                all_metric_rows.append({
                    "group": group,
                    "dataset": name,
                    "threshold_quantile": q,
                    "threshold_label": q_labels[q],
                    "frames": T,
                    "gt_nodes_sparse": int(gt.num_nodes()),
                    "gt_edges": gt_edges,
                    "pred_nodes": int(er.num_pred_nodes),
                    "estimated_total_nodes": total_est,
                    "pred_over_estimated_nodes": int(er.num_pred_nodes) / total_est if total_est else float("nan"),
                    "pred_nodes_per_frame": int(er.num_pred_nodes) / T,
                    "node_recall": float(rec),
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
                    "edge_jaccard": float(pm["edge_jaccard"]),
                    "adj_edge_jaccard": float(pm["adj_edge_jaccard"]),
                })
                del pred
                gc.collect()
            del gt, coords_by_q
            gc.collect()

    with (OUT / "frame_detection_counts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_frame_rows[0].keys()))
        w.writeheader(); w.writerows(all_frame_rows)
    with (OUT / "detection_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detection_manifest[0].keys()))
        w.writeheader(); w.writerows(detection_manifest)
    with (OUT / "detection_metrics_per_dataset.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_metric_rows[0].keys()))
        w.writeheader(); w.writerows(all_metric_rows)

    sweep_rows = []
    for group in groups:
        for q in q_values:
            s = summarise(official_by_group_q[(group, q)])
            subset = [r for r in all_metric_rows if r["group"] == group and r["threshold_quantile"] == q]
            ge = sum(r["gt_edges"] for r in subset)
            re = sum(r["recoverable_gt_edges"] for r in subset)
            tp = sum(r["edge_tp"] for r in subset)
            pred_n = sum(r["pred_nodes"] for r in subset)
            est_n = sum(r["estimated_total_nodes"] for r in subset if math.isfinite(r["estimated_total_nodes"]))
            sweep_rows.append({
                "group": group,
                "threshold_quantile": q,
                "threshold_label": q_labels[q],
                "n_datasets": len(subset),
                "pred_nodes": pred_n,
                "estimated_total_nodes": est_n,
                "pred_over_estimated_nodes": pred_n / est_n if est_n else float("nan"),
                "node_recall": float(s["node_recall"]),
                "edge_endpoint_coverage_micro": re / ge if ge else float("nan"),
                "conditional_link_recall_micro": tp / re if re else float("nan"),
                "edge_jaccard": float(s["edge_jaccard"]),
                "division_jaccard": float(s["division_jaccard"]),
                "adj_edge_jaccard": float(s["adj_edge_jaccard"]),
                "score": float(s["score"]),
                "edge_fp": sum(r["edge_fp"] for r in subset),
                "missing_node_gt_edges": sum(r["missing_node_gt_edges"] for r in subset),
                "link_miss_gt_edges": sum(r["link_miss_gt_edges"] for r in subset),
            })

    with (OUT / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader(); w.writerows(sweep_rows)

    cal_rows = [r for r in sweep_rows if r["group"] == "calibration"]
    best_cal = max(cal_rows, key=lambda r: r["score"])
    selected_q = best_cal["threshold_quantile"]
    confirm = next(r for r in sweep_rows if r["group"] == "confirmation" and r["threshold_quantile"] == selected_q)
    payload = {
        "status": "passed",
        "provenance": "classical_detection_calibration4_confirmation4_no_learned_weights",
        "selected_threshold_quantile_from_calibration": selected_q,
        "calibration_selected": best_cal,
        "confirmation_at_selected_threshold": confirm,
        "all_sweeps": sweep_rows,
        "interpretation": (
            "This is a classical image-processing diagnostic, not learned-model OOF. "
            "The threshold is selected using only the 4-video calibration group and then "
            "read once on the disjoint 4-video confirmation group. Linking is fixed to "
            "10 um one-to-one Hungarian and no divisions are predicted."
        ),
    }
    (OUT / "detection_pilot_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (OUT / "progress.json").write_text(json.dumps({"phase": "complete"}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))

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
    print(json.dumps({"kernel": KERNEL_ID, "github_run_id": os.environ.get("GITHUB_RUN_ID", "local")}, indent=2))


if __name__ == "__main__":
    main()
