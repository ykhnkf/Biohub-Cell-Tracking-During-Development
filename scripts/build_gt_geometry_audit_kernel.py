from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_cv_audit_kernel_v7 import build_worker as build_audit_worker_v7

KERNEL_ID = "ykhnkf/biohub-gt-geometry-audit"
KERNEL_TITLE = "Biohub GT Geometry Audit"


def build_worker() -> str:
    base = build_audit_worker_v7()
    start = base.index("def run() -> None:")
    prefix = base[:start].replace(
        'OUT = Path("/kaggle/working/biohub_cv_audit")',
        'OUT = Path("/kaggle/working/biohub_gt_geometry_audit")', 1,
    )
    run = r'''def run() -> None:
    import gc
    import math
    import numpy as np

    support = find_support_root()
    ensure_dependencies(support)
    copy_vendored_official()
    sys.path.insert(0, str(OFFICIAL / "scripts"))
    sys.path.insert(0, str(OFFICIAL / "src"))

    import tracksdata as td
    from biohub_tracking.io import DEFAULT_SCALE, open_dataset

    gates = [3.0, 5.0, 7.0, 10.0, 15.0]

    def load_graph(path: Path):
        obj = td.graph.IndexedRXGraph.from_geff(path)
        return obj[0] if isinstance(obj, tuple) else obj

    def scale_for(name: str):
        try:
            return np.asarray(open_dataset(TRAIN / name, load_image=False).scale, dtype=np.float64)
        except Exception:
            return np.asarray(DEFAULT_SCALE, dtype=np.float64)

    def q(values, p):
        if not values:
            return float("nan")
        return float(np.quantile(np.asarray(values, dtype=float), p))

    geffs = sorted(TRAIN.glob("*.geff"))
    all_records = []
    per_dataset = []
    for idx, path in enumerate(geffs):
        name = path.stem
        g = load_graph(path)
        scale = scale_for(name)
        node_df = g.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        edge_df = g.edge_attrs(attr_keys=["source_id", "target_id"])
        rows = node_df.to_dicts()
        pos = {int(r["node_id"]): np.asarray([r["z"], r["y"], r["x"]], dtype=np.float64) for r in rows}
        tmap = {int(r["node_id"]): int(r["t"]) for r in rows}
        by_t = {}
        for r in rows:
            by_t.setdefault(int(r["t"]), []).append(int(r["node_id"]))
        outdeg = {int(n): int(d) for n, d in zip(g.node_ids(), g.out_degree(g.node_ids()))}
        ds = []
        rank1 = 0
        eligible = 0
        ambiguous = 0
        nondiv_rank1 = 0
        nondiv_n = 0
        div_rank1 = 0
        div_n = 0
        gate_hits = {gate: 0 for gate in gates}
        gate_ambiguous = {gate: 0 for gate in gates}
        for erow in edge_df.to_dicts():
            s = int(erow["source_id"]); t = int(erow["target_id"])
            if s not in pos or t not in pos:
                continue
            dt = tmap[t] - tmap[s]
            dist = float(np.linalg.norm((pos[t] - pos[s]) * scale))
            ds.append(dist)
            rec = {"dataset": name, "source_id": s, "target_id": t, "dt": dt, "distance_um": dist, "source_outdegree": outdeg.get(s, 0)}
            for gate in gates:
                if dist <= gate:
                    gate_hits[gate] += 1
            if dt == 1 and (tmap[s] + 1) in by_t:
                candidates = by_t[tmap[s] + 1]
                dvec = np.asarray([float(np.linalg.norm((pos[c] - pos[s]) * scale)) for c in candidates])
                order = np.argsort(dvec)
                true_index = candidates.index(t)
                true_rank = int(np.where(order == true_index)[0][0]) + 1
                rec["true_target_rank"] = true_rank
                rec["nearest_distance_um"] = float(dvec[order[0]])
                rec["second_distance_um"] = float(dvec[order[1]]) if len(order) > 1 else float("nan")
                rec["candidate_count_7um"] = int(np.sum(dvec <= 7.0))
                eligible += 1
                if true_rank == 1:
                    rank1 += 1
                if int(np.sum(dvec <= 7.0)) >= 2:
                    ambiguous += 1
                if outdeg.get(s, 0) >= 2:
                    div_n += 1
                    div_rank1 += int(true_rank == 1)
                else:
                    nondiv_n += 1
                    nondiv_rank1 += int(true_rank == 1)
                for gate in gates:
                    if int(np.sum(dvec <= gate)) >= 2:
                        gate_ambiguous[gate] += 1
            all_records.append(rec)
        per_dataset.append({
            "dataset": name,
            "n_edges": len(ds),
            "edge_disp_p50_um": q(ds, .5),
            "edge_disp_p90_um": q(ds, .9),
            "edge_disp_p95_um": q(ds, .95),
            "edge_disp_p99_um": q(ds, .99),
            "rank1_eligible_edges": eligible,
            "nearest_neighbor_true_target_rate": rank1 / eligible if eligible else float("nan"),
            "ambiguous_7um_rate": ambiguous / eligible if eligible else float("nan"),
            "nondivision_nn_rate": nondiv_rank1 / nondiv_n if nondiv_n else float("nan"),
            "division_edge_nn_rate": div_rank1 / div_n if div_n else float("nan"),
            **{f"within_{gate:g}um_rate": gate_hits[gate] / len(ds) if ds else float("nan") for gate in gates},
            **{f"multi_candidate_{gate:g}um_rate": gate_ambiguous[gate] / eligible if eligible else float("nan") for gate in gates},
        })
        del g
        gc.collect()
        if (idx + 1) % 20 == 0 or idx + 1 == len(geffs):
            (OUT / "progress.json").write_text(json.dumps({"phase":"geometry","done":idx+1,"total":len(geffs)}, indent=2), encoding="utf-8")

    with (OUT / "geometry_per_dataset.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_dataset[0].keys()))
        w.writeheader(); w.writerows(per_dataset)

    dists = [r["distance_um"] for r in all_records]
    rank_records = [r for r in all_records if "true_target_rank" in r]
    nondiv = [r for r in rank_records if r["source_outdegree"] < 2]
    div = [r for r in rank_records if r["source_outdegree"] >= 2]
    summary = {
        "status": "passed",
        "n_datasets": len(geffs),
        "n_edges": len(dists),
        "edge_displacement_um": {"p50":q(dists,.5),"p75":q(dists,.75),"p90":q(dists,.9),"p95":q(dists,.95),"p99":q(dists,.99),"max":max(dists) if dists else float("nan")},
        "within_gate_fraction": {f"{gate:g}": float(np.mean(np.asarray(dists) <= gate)) for gate in gates},
        "adjacent_edge_rank_audit": {
            "n_edges": len(rank_records),
            "true_target_is_nearest_fraction": float(np.mean([r["true_target_rank"] == 1 for r in rank_records])) if rank_records else float("nan"),
            "true_target_in_top2_fraction": float(np.mean([r["true_target_rank"] <= 2 for r in rank_records])) if rank_records else float("nan"),
            "true_target_in_top3_fraction": float(np.mean([r["true_target_rank"] <= 3 for r in rank_records])) if rank_records else float("nan"),
            "multi_candidate_7um_fraction": float(np.mean([r["candidate_count_7um"] >= 2 for r in rank_records])) if rank_records else float("nan"),
            "nondivision_true_target_is_nearest_fraction": float(np.mean([r["true_target_rank"] == 1 for r in nondiv])) if nondiv else float("nan"),
            "division_edge_true_target_is_nearest_fraction": float(np.mean([r["true_target_rank"] == 1 for r in div])) if div else float("nan"),
            "n_nondivision_edges": len(nondiv),
            "n_division_edges": len(div),
        },
        "interpretation": "GT-only structural diagnostic. It measures physical motion and ambiguity among annotated next-frame nodes; it is not a model score and sparse annotation can understate real image-level crowding.",
    }
    (OUT / "geometry_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (OUT / "progress.json").write_text(json.dumps({"phase":"complete","n_datasets":len(geffs)}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))

try:
    run()
except Exception as exc:
    payload={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
    (OUT/"fatal_error.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2)); raise
'''
    return prefix + run


def main() -> None:
    out = Path(".kaggle_gt_geometry_audit_kernel")
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "run_geometry_audit.py").write_text(build_worker(), encoding="utf-8")
    meta = {
        "id": KERNEL_ID, "title": KERNEL_TITLE, "code_file": "run_geometry_audit.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "keywords": [], "dataset_sources": [SUPPORT_DATASET], "kernel_sources": [],
        "competition_sources": [COMPETITION], "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"kernel": KERNEL_ID, "github_run_id": os.environ.get("GITHUB_RUN_ID", "local")}, indent=2))


if __name__ == "__main__": main()
