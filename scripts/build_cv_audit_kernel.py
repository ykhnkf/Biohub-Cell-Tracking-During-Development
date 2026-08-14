from __future__ import annotations

import json
from pathlib import Path

KERNEL_ID = "ykhnkf/biohub-cell-tracking-cv-audit"
KERNEL_TITLE = "Biohub Cell Tracking CV Audit"
COMPETITION = "biohub-cell-tracking-during-development"
# Freeze the evaluator implementation used by the first CV benchmark so future
# upstream changes cannot silently move our local score.
OFFICIAL_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"


def build_worker() -> str:
    return rf'''from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

OUT = Path("/kaggle/working/biohub_cv_audit")
OUT.mkdir(parents=True, exist_ok=True)
TRAIN = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train")
OFFICIAL = Path("/kaggle/working/official")
OFFICIAL_COMMIT = "{OFFICIAL_COMMIT}"

# Pin the public official evaluator to an exact commit for reproducibility.
subprocess.run(["git", "init", str(OFFICIAL)], check=True)
subprocess.run(["git", "-C", str(OFFICIAL), "remote", "add", "origin", "https://github.com/royerlab/kaggle-cell-tracking-competition.git"], check=True)
subprocess.run(["git", "-C", str(OFFICIAL), "fetch", "--depth", "1", "origin", OFFICIAL_COMMIT], check=True)
subprocess.run(["git", "-C", str(OFFICIAL), "checkout", "--detach", "FETCH_HEAD"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(OFFICIAL)], check=True)
sys.path.insert(0, str(OFFICIAL / "scripts"))
sys.path.insert(0, str(OFFICIAL / "src"))

import tracksdata as td
from geff import GeffMetadata
from tracking_cellmot.io import DEFAULT_SCALE, open_dataset
from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise


def load_graph(path: Path):
    obj = td.graph.IndexedRXGraph.from_geff(path)
    return obj[0] if isinstance(obj, tuple) else obj


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

rows = []
metric_rows = []
selftest_failures = []

for path in geffs:
    name = path.stem
    graph = load_graph(path)
    n_nodes = int(graph.num_nodes())
    n_edges = int(graph.num_edges())
    n_div = divisions(graph)
    n_total = estimated_total(path)
    scale = dataset_scale(name)

    # GT -> GT must be perfect (up to any explicit node-count adjustment metadata).
    pred = load_graph(path)
    gt = load_graph(path)
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

if not rows:
    raise RuntimeError(f"No ground-truth .geff files found below {{TRAIN}}")

fields = list(rows[0].keys())
with (OUT / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# Primary split candidate: leave one complete dataset/embryo out.
names = [r["dataset"] for r in rows]
lodo = [
    {{"fold": i, "train": [x for x in names if x != holdout], "valid": [holdout]}}
    for i, holdout in enumerate(names)
]
(OUT / "folds_lodo.json").write_text(json.dumps(lodo, indent=2), encoding="utf-8")

# Secondary split candidate: deterministic greedy 5-fold balancing the units
# most relevant to the official score (edges first, divisions second, nodes third).
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

versions = {{
    "python": sys.version,
    "platform": platform.platform(),
    "official_commit": OFFICIAL_COMMIT,
    "tracksdata": getattr(td, "__version__", "unknown"),
}}
(OUT / "audit_environment.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
'''


def main() -> None:
    out = Path(".kaggle_cv_audit_kernel")
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
        # Audit/OOF development kernel, not a competition submission kernel.
        "enable_internet": True,
        "keywords": ["biohub", "cell-tracking", "cross-validation", "audit"],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"kernel": KERNEL_ID, "out": str(out), "official_commit": OFFICIAL_COMMIT}, indent=2))


if __name__ == "__main__":
    main()
