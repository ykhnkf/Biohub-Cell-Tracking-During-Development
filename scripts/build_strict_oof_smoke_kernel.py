from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_cv_audit_kernel_v7 import build_worker as build_audit_worker_v7

KERNEL_ID = "ykhnkf/biohub-strict-oof-smoke"
KERNEL_TITLE = "Biohub Strict OOF Train Predict Score Smoke"

TRAIN_NAMES = [
    '44b6_0b24845f', '44b6_18ced818', '44b6_87bba6c4', '44b6_66f9292d',
    '44b6_551a5dba', '44b6_415c0a3a', '44b6_7a302da0', '44b6_3a861e03',
    '6bba_0e7c0d07', '6bba_f1fde7e0', '6bba_312f0dc3', '6bba_b1ae37b9',
    '6bba_718b21f9', '6bba_789f8168', '6bba_2646afc7', '6bba_fc516dc6',
    '6bba_78a7bd97', '6bba_062c8d37', '6bba_b693381b', '6bba_f17befbc',
    '6bba_5c039895', '6bba_b329af44', '6bba_1d0d8384', '6bba_09961292',
]
VAL_NAMES = ['44b6_81c256f0', '44b6_9be80b04', '6bba_474be664', '6bba_6321a359']


def build_worker() -> str:
    base = build_audit_worker_v7()
    start = base.index("def run() -> None:")
    prefix = base[:start].replace(
        'OUT = Path("/kaggle/working/biohub_cv_audit")',
        'OUT = Path("/kaggle/working/biohub_strict_oof_smoke")', 1,
    )
    train_names = repr(TRAIN_NAMES)
    val_names = repr(VAL_NAMES)
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
    from tracksdata.metrics import DistanceMatching
    from geff import GeffMetadata
    from biohub_tracking.io import DEFAULT_SCALE, open_dataset, save_graph
    from biohub_tracking.metrics import evaluate, node_recall, per_sample_metrics, summarise
    from predict_unet_transformer import PredictConfig, build_graph, predict_video
    from train_unet_transformer import train

    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    device = torch.device("cuda")
    train_names = __TRAIN_NAMES__
    val_names = __VAL_NAMES__
    overlap = sorted(set(train_names) & set(val_names))
    if overlap:
        raise RuntimeError(f"Train/validation leakage in declared split: {overlap}")
    missing = [n for n in train_names + val_names if not (TRAIN / f"{n}.zarr").exists() or not (TRAIN / f"{n}.geff").exists()]
    if missing:
        raise FileNotFoundError(f"Missing datasets: {missing}")

    split_path = OUT / "smoke_splits.json"
    split_path.write_text(json.dumps([{"split":0,"train":train_names,"test":val_names}], indent=2), encoding="utf-8")
    (OUT / "split_manifest.json").write_text(json.dumps({
        "provenance":"strict_oof_smoke_random_init",
        "train":train_names,"validation":val_names,
        "overlap":overlap,
        "n_train":len(train_names),"n_validation":len(val_names),
    }, indent=2), encoding="utf-8")

    method = "strict_oof_smoke"
    (OUT / "progress.json").write_text(json.dumps({"phase":"training"}, indent=2), encoding="utf-8")
    t0 = time.time()
    model = train(
        data_dir=TRAIN,
        fold=0,
        splits_file=split_path,
        method=method,
        n_epochs=1,
        lr=1e-4,
        batch_size=1,
        num_workers=0,
        unet_out_channels=32,
        unet_layers=[32,64,128],
        unet_weights=None,
        downsample=(1,4,4),
        det_loss_weight=1.0,
        det_neg_weight=1e-2,
        max_iters=60,
        seed=20260815,
        window_size=2,
        pool_kernel_um=5.0,
        data_parallel=False,
    )
    train_seconds = time.time() - t0
    model.to(device); model.eval()

    weights_dir = OFFICIAL / "weights" / method / "split_0"
    weight_files = [str(p) for p in sorted(weights_dir.glob("*")) if p.is_file()]
    (OUT / "training_summary.json").write_text(json.dumps({
        "train_seconds":train_seconds,
        "n_epochs":1,"max_iters":60,"batch_size":1,"num_workers":0,"downsample":[1,4,4],
        "weights_dir":str(weights_dir),"weight_files":weight_files,
    }, indent=2), encoding="utf-8")

    cfg = PredictConfig(
        det_threshold=0.5,
        det_tta=False,
        pool_kernel_um=5.0,
        edge_activation="softmax",
        threshold=0.5,
        use_ilp=False,
    )
    pred_dir = OUT / "predictions"; pred_dir.mkdir(parents=True, exist_ok=True)
    pred_manifest=[]
    for name in val_names:
        (OUT / "progress.json").write_text(json.dumps({"phase":"predicting","dataset":name,"completed":[x["dataset"] for x in pred_manifest]}, indent=2), encoding="utf-8")
        t1=time.time()
        coords, edges = predict_video(model, TRAIN / f"{name}.zarr", device, cfg=cfg, window_size=2, downsample=(1,4,4))
        graph=build_graph(coords,edges)
        save_graph(graph,pred_dir/f"{name}.geff")
        pred_manifest.append({"dataset":name,"pred_nodes":int(graph.num_nodes()),"pred_edges":int(graph.num_edges()),"seconds":time.time()-t1})
        del coords,edges,graph; gc.collect(); torch.cuda.empty_cache()

    with (OUT/"prediction_manifest.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(pred_manifest[0])); w.writeheader(); w.writerows(pred_manifest)

    def load_graph(path: Path):
        obj=td.graph.IndexedRXGraph.from_geff(path); return obj[0] if isinstance(obj,tuple) else obj
    def estimated_total(path: Path):
        meta=GeffMetadata.read(path); v=(meta.extra or {}).get("estimated_number_of_nodes"); return float(v) if v is not None else float("nan")
    def scale(name):
        try: return tuple(float(x) for x in open_dataset(TRAIN/name,load_image=False).scale)
        except Exception: return tuple(float(x) for x in DEFAULT_SCALE)
    def ensure_node_matching(pred, gt, sc):
        """Official evaluate() returns early for edge-less predictions before graph.match().

        node_recall() requires MATCHED_NODE_ID to already exist, so explicitly
        run the same DistanceMatching node match for this boundary case.
        """
        k=td.DEFAULT_ATTR_KEYS
        if pred.num_nodes() == 0:
            return False
        if k.MATCHED_NODE_ID not in pred.node_attr_keys():
            pred.match(gt, matching=DistanceMatching(max_distance=7.0, scale=sc))
        return k.MATCHED_NODE_ID in pred.node_attr_keys()
    def safe_node_recall(pred, gt, sc):
        if pred.num_nodes() == 0 or gt.num_nodes() == 0:
            return 0.0
        if not ensure_node_matching(pred, gt, sc):
            return 0.0
        return float(node_recall(pred,gt))
    def matched_gt(pred):
        k=td.DEFAULT_ATTR_KEYS
        if k.MATCHED_NODE_ID not in pred.node_attr_keys():
            return set()
        a=pred.node_attrs(attr_keys=[k.MATCHED_NODE_ID]); return {int(r[k.MATCHED_NODE_ID]) for r in a.to_dicts() if int(r[k.MATCHED_NODE_ID])>=0}
    def recoverable(gt,matched):
        k=td.DEFAULT_ATTR_KEYS; a=gt.edge_attrs(attr_keys=[k.EDGE_SOURCE,k.EDGE_TARGET]); return sum(int(r[k.EDGE_SOURCE]) in matched and int(r[k.EDGE_TARGET]) in matched for r in a.to_dicts())

    rows=[]; official_rows=[]
    for info in pred_manifest:
        name=info["dataset"]; pred=load_graph(pred_dir/f"{name}.geff"); gt=load_graph(TRAIN/f"{name}.geff")
        sc=scale(name)
        er=evaluate(pred,gt,scale=sc,max_distance=7.0)
        rec=safe_node_recall(pred,gt,sc)
        total=estimated_total(TRAIN/f"{name}.geff")
        pm=per_sample_metrics(er,total,rec); official_rows.append(pm); matched=matched_gt(pred); recov=recoverable(gt,matched); nge=int(gt.num_edges())
        rows.append({
            "dataset":name,"gt_nodes_sparse":int(gt.num_nodes()),"gt_edges":nge,"pred_nodes":int(er.num_pred_nodes),"estimated_total_nodes":total,
            "node_recall":float(rec),"recoverable_gt_edges":int(recov),"edge_endpoint_coverage":recov/nge if nge else float("nan"),
            "edge_tp":int(er.edge_tp),"edge_fp":int(er.edge_fp),"edge_fn":int(er.edge_fn),"conditional_link_recall":int(er.edge_tp)/recov if recov else float("nan"),
            "missing_node_gt_edges":max(0,nge-recov),"link_miss_gt_edges":max(0,recov-int(er.edge_tp)),
            "division_tp":int(er.division_tp),"division_fp":int(er.division_fp),"division_fn":int(er.division_fn),
            "edge_jaccard":float(pm["edge_jaccard"]),"adj_edge_jaccard":float(pm["adj_edge_jaccard"]),
        })
        del pred,gt; gc.collect()

    with (OUT/"oof_smoke_per_dataset.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary=summarise(official_rows); ge=sum(r["gt_edges"] for r in rows); re=sum(r["recoverable_gt_edges"] for r in rows); tp=sum(r["edge_tp"] for r in rows)
    payload={
        "status":"passed","provenance":"strict_oof_smoke_random_init_no_train_val_overlap",
        "official":{k:(int(v) if isinstance(v,int) else float(v)) for k,v in summary.items()},
        "diagnostics":{
            "edge_endpoint_coverage_micro":re/ge if ge else float("nan"),"conditional_link_recall_micro":tp/re if re else float("nan"),
            "missing_node_gt_edges":sum(r["missing_node_gt_edges"] for r in rows),"link_miss_gt_edges":sum(r["link_miss_gt_edges"] for r in rows),"edge_fp":sum(r["edge_fp"] for r in rows),
        },
        "training":{"n_train":len(train_names),"n_validation":len(val_names),"epochs":1,"max_iters":60,"num_workers":0,"random_init":True,"train_seconds":train_seconds},
        "inference_config":{"det_threshold":0.5,"det_tta":False,"pool_kernel_um":5.0,"edge_threshold":0.5},
    }
    (OUT/"oof_smoke_summary.json").write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8")
    (OUT/"progress.json").write_text(json.dumps({"phase":"complete"},indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2,default=str))

try:
    run()
except Exception as exc:
    payload={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}; (OUT/"fatal_error.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2)); raise
'''
    return prefix + run.replace("__TRAIN_NAMES__", train_names).replace("__VAL_NAMES__", val_names)


def main() -> None:
    out=Path(".kaggle_strict_oof_smoke_kernel")
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    (out/"run_strict_oof_smoke.py").write_text(build_worker(),encoding="utf-8")
    meta={"id":KERNEL_ID,"title":KERNEL_TITLE,"code_file":"run_strict_oof_smoke.py","language":"python","kernel_type":"script","is_private":True,"enable_gpu":True,"enable_tpu":False,"enable_internet":False,"keywords":[],"dataset_sources":[SUPPORT_DATASET],"kernel_sources":[],"competition_sources":[COMPETITION],"model_sources":[]}
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps({"kernel":KERNEL_ID,"github_run_id":os.environ.get("GITHUB_RUN_ID","local"),"n_train":len(TRAIN_NAMES),"n_val":len(VAL_NAMES)},indent=2))

if __name__=="__main__": main()
