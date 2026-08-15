from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, KERNEL_ID, KERNEL_TITLE, SUPPORT_DATASET
from build_cv_audit_kernel_v6 import build_worker as build_worker_v6


def build_worker() -> str:
    worker = build_worker_v6()
    start_marker = "    # Full metric matching is deliberately deferred until the manifest is known."
    end_marker = "    edge_counts = [r[\"gt_edges\"] for r in rows]"
    start = worker.index(start_marker)
    end = worker.index(end_marker, start)

    targeted_selftest = r'''    # Run one real GT->GT evaluation through the official metric implementation.
    # Use the smallest annotated graph that contains at least one division, so
    # both edge and division paths are exercised without repeating the v5
    # all-dataset self-match that was externally cancelled on Kaggle.
    from biohub_tracking.metrics import evaluate, node_recall, per_sample_metrics, summarise

    candidates = [
        r for r in rows
        if r["gt_edges"] > 0 and r["gt_divisions"] > 0
    ]
    if not candidates:
        raise RuntimeError("No loaded dataset contains a division for metric self-test")

    target_row = min(
        candidates,
        key=lambda r: (r["gt_nodes"], r["gt_edges"], r["dataset"]),
    )
    target_path = TRAIN / f"{target_row['dataset']}.geff"
    pred = load_graph(td, target_path)
    gt = load_graph(td, target_path)
    er = evaluate(
        pred,
        gt,
        scale=(target_row["scale_z"], target_row["scale_y"], target_row["scale_x"]),
        max_distance=7.0,
    )
    recall = node_recall(pred, gt)
    sample_metrics = per_sample_metrics(
        er,
        float(target_row["estimated_number_of_nodes"]),
        recall,
    )
    aggregate = summarise([sample_metrics])

    edge_identity_ok = (
        int(er.edge_tp) == int(target_row["gt_edges"])
        and int(er.edge_fp) == 0
        and int(er.edge_fn) == 0
        and float(recall) > 0.999999
    )
    division_identity_ok = (
        int(er.division_tp) == int(target_row["gt_divisions"])
        and int(er.division_fp) == 0
        and int(er.division_fn) == 0
    )
    metric_selftest = {
        "status": "passed" if edge_identity_ok and division_identity_ok else "failed",
        "dataset": target_row["dataset"],
        "gt_nodes": int(target_row["gt_nodes"]),
        "gt_edges": int(target_row["gt_edges"]),
        "gt_divisions": int(target_row["gt_divisions"]),
        "estimated_number_of_nodes": float(target_row["estimated_number_of_nodes"]),
        "scale": [target_row["scale_z"], target_row["scale_y"], target_row["scale_x"]],
        "evaluation_result": {k: int(v) for k, v in er._asdict().items()},
        "node_recall": float(recall),
        "per_sample_metrics": {k: float(v) for k, v in sample_metrics.items()},
        "summary": {
            k: (int(v) if isinstance(v, int) else float(v))
            for k, v in aggregate.items()
        },
        "edge_identity_ok": bool(edge_identity_ok),
        "division_identity_ok": bool(division_identity_ok),
        "metric_module": str(Path(metric_module.__file__).resolve()),
    }
    (OUT / "official_metric_selftest.json").write_text(
        json.dumps(metric_selftest, indent=2), encoding="utf-8"
    )
    if metric_selftest["status"] != "passed":
        raise RuntimeError(
            "Official metric GT-to-GT self-test failed; see official_metric_selftest.json"
        )

'''
    worker = worker[:start] + targeted_selftest + worker[end:]
    worker = worker.replace(
        '"metric_selftest": "import_only_phase0",',
        '"metric_selftest": metric_selftest["status"],',
        1,
    )

    # Stamp every Kaggle output with the originating GitHub Actions run ID.
    # This makes a stale output from an earlier kernel version impossible to
    # accept as a successful current run.
    run_token = os.environ.get("GITHUB_RUN_ID", "local")
    needle = 'OUT.mkdir(parents=True, exist_ok=True)\n'
    injection = needle + (
        f'AUDIT_RUN_TOKEN = {run_token!r}\n'
        '(OUT / "run_token.json").write_text(json.dumps({"github_run_id": AUDIT_RUN_TOKEN}, indent=2), encoding="utf-8")\n'
    )
    if needle not in worker:
        raise RuntimeError("Could not locate OUT initialization for run-token injection")
    worker = worker.replace(needle, injection, 1)
    return worker


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
        "mode": "manifest-plus-targeted-metric-selftest-v7",
        "github_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
    }, indent=2))


if __name__ == "__main__":
    main()
