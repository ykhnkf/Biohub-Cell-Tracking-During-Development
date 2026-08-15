from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_cv_audit_kernel_v7 import build_worker as build_audit_worker_v7

KERNEL_ID = "ykhnkf/biohub-linking-oracle-audit"
KERNEL_TITLE = "Biohub GT Node Linking Oracle Audit"


def build_worker() -> str:
    base = build_audit_worker_v7()
    start = base.index("def run() -> None:")
    prefix = base[:start].replace(
        'OUT = Path("/kaggle/working/biohub_cv_audit")',
        'OUT = Path("/kaggle/working/biohub_linking_oracle")', 1,
    )
    run = r'''def run() -> None:
    import gc
    import math
    import numpy as np
    import polars as pl
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

    gates = [3.0, 5.0, 7.0, 10.0, 15.0]

    def load_graph(path: Path):
        obj = td.graph.IndexedRXGraph.from_geff(path)
        return obj[0] if isinstance(obj, tuple) else obj

    def scale_for(name: str):
        try:
            return np.asarray(open_dataset(TRAIN / name, load_image=False).scale, dtype=np.float64)
        except Exception:
            return np.asarray(DEFAULT_SCALE, dtype=np.float64)

    def estimated_total(path: Path) -> float:
        try:
            meta = GeffMetadata.read(path)
            v = (meta.extra or {}).get("estimated_number_of_nodes")
            return float(v) if v is not None else float("nan")
        except Exception:
            return float("nan")

    def graph_nodes(gt):
        return gt.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"]).sort(["t", "node_id"])

    def true_edge_distances(gt, attrs, scale):
        coords = {int(r["node_id"]): np.asarray([r["z"], r["y"], r["x"]], dtype=np.float64) for r in attrs.to_dicts()}
        e = gt.edge_attrs(attr_keys=["source_id", "target_id"])
        out = []
        for r in e.to_dicts():
            s, t = int(r["source_id"]), int(r["target_id"])
            if s in coords and t in coords:
                out.append(float(np.linalg.norm((coords[t] - coords[s]) * scale)))
        return out

    def make_pred(attrs, scale, gate):
        g = td.graph.InMemoryGraph()
        for key in ["z", "y", "x"]:
            g.add_node_attr_key(key, pl.Float64, -999999.0)
        rows = attrs.to_dicts()
        new_ids = g.bulk_add_nodes([{"t":int(r["t"]),"z":float(r["z"]),"y":float(r["y"]),"x":float(r["x"])} for r in rows])
        by_t = {}
        for i, r in enumerate(rows):
            by_t.setdefault(int(r["t"]), []).append(i)
        edge_rows = []
        for t in sorted(by_t):
            if t + 1 not in by_t:
                continue
            ia, ib = by_t[t], by_t[t+1]
            A = np.asarray([[rows[i]["z"], rows[i]["y"], rows[i]["x"]] for i in ia], dtype=np.float64)
            B = np.asarray([[rows[i]["z"], rows[i]["y"], rows[i]["x"]] for i in ib], dtype=np.float64)
            if len(A) == 0 or len(B) == 0:
                continue
            D = np.linalg.norm((A[:,None,:] - B[None,:,:]) * scale[None,None,:], axis=2)
            cost = D.copy()
            bad = D > gate
            cost[bad] = gate + 1e6 + D[bad]
            rr, cc = linear_sum_assignment(cost)
            for rix, cix in zip(rr, cc):
                d = float(D[rix,cix])
                if d <= gate:
                    edge_rows.append({"source_id":new_ids[ia[rix]],"target_id":new_ids[ib[cix]]})
        if edge_rows:
            g.bulk_add_edges(edge_rows)
        return g

    geffs = sorted(TRAIN.glob("*.geff"))
    if not geffs:
        raise RuntimeError("No train GT geffs found")

    all_dists = []
    prefix_dists = {}
    rows_by_gate = {g: [] for g in gates}
    for idx, path in enumerate(geffs):
        name = path.stem
        gt = load_graph(path)
        attrs = graph_nodes(gt)
        scale = scale_for(name)
        dists = true_edge_distances(gt, attrs, scale)
        all_dists.extend(dists)
        prefix_dists.setdefault(name[:4], []).extend(dists)
        total = estimated_total(path)
        for gate in gates:
            pred = make_pred(attrs, scale, gate)
            er = evaluate(pred, gt, scale=tuple(scale), max_distance=7.0)
            rec = node_recall(pred, gt)
            pm = per_sample_metrics(er, total, rec)
            rows_by_gate[gate].append({"dataset":name, **pm})
            del pred
        del gt
        gc.collect()
        if (idx + 1) % 20 == 0 or idx + 1 == len(geffs):
            (OUT / "progress.json").write_text(json.dumps({"phase":"evaluating","done":idx+1,"total":len(geffs)}, indent=2), encoding="utf-8")

    def qstats(values):
        a = np.asarray(values, dtype=float)
        if a.size == 0: return {}
        qs = [0.5,0.75,0.9,0.95,0.99,0.995,1.0]
        return {f"p{int(q*1000)/10:g}":float(np.quantile(a,q)) for q in qs}

    displacement = {"all":qstats(all_dists), "by_prefix":{k:qstats(v) for k,v in prefix_dists.items()}}
    (OUT / "gt_edge_displacement_um.json").write_text(json.dumps(displacement, indent=2), encoding="utf-8")

    gate_summary = []
    for gate in gates:
        s = summarise(rows_by_gate[gate])
        candidate_recall = float(np.mean(np.asarray(all_dists) <= gate)) if all_dists else float("nan")
        row = {"gate_um":gate,"gt_edge_within_gate_fraction":candidate_recall, **{k:(int(v) if isinstance(v,int) else float(v)) for k,v in s.items()}}
        gate_summary.append(row)
    with (OUT / "gate_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(gate_summary[0])); w.writeheader(); w.writerows(gate_summary)

    best = max(gate_summary, key=lambda r: r["edge_jaccard"] if math.isfinite(r["edge_jaccard"]) else -1)
    payload={
        "status":"passed","provenance":"gt_node_oracle_geometry_only_not_model_cv","n_datasets":len(geffs),
        "n_gt_edges":len(all_dists),"displacement_um":displacement,"gates":gate_summary,
        "best_gate_by_edge_jaccard":best,
        "interpretation":"All GT node coordinates are supplied to the prediction graph; only temporal edges are reconstructed by one-to-one Hungarian distance matching. This isolates association difficulty and cannot be interpreted as an end-to-end score.",
    }
    (OUT / "linking_oracle_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (OUT / "progress.json").write_text(json.dumps({"phase":"complete","n_datasets":len(geffs)}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))

try:
    run()
except Exception as exc:
    payload={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}; (OUT/"fatal_error.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2)); raise
'''
    return prefix + run


def main() -> None:
    out=Path(".kaggle_linking_oracle_kernel")
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    (out/"run_linking_oracle.py").write_text(build_worker(),encoding="utf-8")
    meta={"id":KERNEL_ID,"title":KERNEL_TITLE,"code_file":"run_linking_oracle.py","language":"python","kernel_type":"script","is_private":True,"enable_gpu":False,"enable_tpu":False,"enable_internet":False,"keywords":[],"dataset_sources":[SUPPORT_DATASET],"kernel_sources":[],"competition_sources":[COMPETITION],"model_sources":[]}
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps({"kernel":KERNEL_ID,"github_run_id":os.environ.get("GITHUB_RUN_ID","local")},indent=2))

if __name__=="__main__": main()
