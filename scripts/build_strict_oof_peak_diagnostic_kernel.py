from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_cv_audit_kernel_v7 import build_worker as build_audit_worker_v7
from build_strict_oof_smoke_kernel import TRAIN_NAMES, VAL_NAMES

KERNEL_ID = "ykhnkf/biohub-strict-oof-peak-diagnostic"
KERNEL_TITLE = "Biohub Strict OOF Peak Diagnostic"


def build_worker() -> str:
    base=build_audit_worker_v7()
    start=base.index("def run() -> None:")
    prefix=base[:start].replace('OUT = Path("/kaggle/working/biohub_cv_audit")','OUT = Path("/kaggle/working/biohub_strict_oof_peak_diagnostic")',1)
    # Remove recursive support fallback from inherited helpers.
    fallback='''    root = Path("/kaggle/input")\n    if root.exists():\n        for manifest in root.rglob("ARTIFACT_MANIFEST.json"):\n            if "biohub-tracking-support-pack-50ep-v1" in str(manifest):\n                return manifest.parent\n    raise FileNotFoundError("Attached biohub-tracking-support-pack-50ep-v1 was not found")\n'''
    if fallback in prefix:
        prefix=prefix.replace(fallback,'    raise FileNotFoundError("Support pack not found at known direct paths")\n',1)
    run=r'''def run() -> None:
    import gc
    import numpy as np
    import torch

    train_names=__TRAIN__
    val_names=__VAL__
    roots=[Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),Path("/kaggle/input/biohub-cell-tracking-during-development/train")]
    train_root=next((r for r in roots if all((r/f"{n}.geff").exists() and (r/f"{n}.zarr").exists() for n in train_names+val_names)),None)
    if train_root is None: raise FileNotFoundError("Could not resolve train root")
    support=next((p for p in SUPPORT_CANDIDATES if p.exists()),None)
    if support is None: raise FileNotFoundError("Support pack missing")
    ensure_dependencies(support); copy_vendored_official()
    sys.path.insert(0,str(OFFICIAL/"scripts")); sys.path.insert(0,str(OFFICIAL/"src"))
    import train_unet_transformer as tut

    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    (OUT/"stage_00_ready.json").write_text(json.dumps({"gpu":torch.cuda.get_device_name(0),"train_root":str(train_root)},indent=2),encoding="utf-8")

    split_path=OUT/"peak_diag_splits.json"
    split_path.write_text(json.dumps([{"split":0,"train":train_names,"test":val_names}],indent=2),encoding="utf-8")

    # train() normally runs a full detection+edge validation after the epoch.
    # Stub only that internal model-selection pass so we can inspect detections safely.
    original_evaluate=tut.evaluate
    tut.evaluate=lambda model, loader, device, pool_kernel_um=5.0: (0.0,1.0,1.0)
    (OUT/"stage_01_before_train.json").write_text(json.dumps({"max_iters":60,"internal_evaluate":"stubbed"},indent=2),encoding="utf-8")
    model=tut.train(data_dir=train_root,fold=0,splits_file=split_path,method="strict_oof_peak_diag",n_epochs=1,lr=1e-4,batch_size=1,num_workers=0,unet_out_channels=32,unet_layers=[32,64,128],unet_weights=None,downsample=(1,4,4),det_loss_weight=1.0,det_neg_weight=1e-2,max_iters=60,seed=20260815,window_size=2,pool_kernel_um=5.0,data_parallel=False)
    tut.evaluate=original_evaluate
    model.eval(); device=torch.device("cuda:0")
    (OUT/"stage_02_train_returned.json").write_text(json.dumps({"status":"ok"},indent=2),encoding="utf-8")

    rows=[]
    for name in val_names:
        vm,windows=tut.load_dataset_windows(train_root/name,window_size=2,downsample=(1,4,4))
        ds=tut.FrameWindowDataset([(vm,windows)],augmentations=[])
        loader=torch.utils.data.DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)
        frame_counts=[]; pair_products=[]; worst=[]
        for wi,batch in enumerate(loader):
            imgs=batch["imgs"].to(device,dtype=torch.float32)
            coords=batch["coords"].to(device); masks=batch["masks"].to(device)
            image_shape=tuple(batch["image_shape"][0].tolist()); voxel_size=tuple(batch["voxel_size"][0].tolist())
            with torch.no_grad():
                _,det_logits=model.encode(imgs)
                ns=[]
                for fi in range(2):
                    dc,dp,dm,matches=tut.detect_and_match(det_logits[fi],coords[:,fi],masks[:,fi],image_shape,voxel_size=voxel_size,pool_kernel_um=5.0,frame_index=fi,window_size=2)
                    n=int(dm[0].sum().item()); ns.append(n); frame_counts.append(n)
                prod=int(ns[0]*ns[1]); pair_products.append(prod)
                worst.append((prod,wi,ns[0],ns[1]))
            del imgs,coords,masks,det_logits; torch.cuda.empty_cache()
        a=np.asarray(frame_counts,dtype=np.int64); p=np.asarray(pair_products,dtype=np.int64); worst.sort(reverse=True)
        row={"dataset":name,"n_windows":len(windows),"peak_min":int(a.min()),"peak_p50":float(np.percentile(a,50)),"peak_p90":float(np.percentile(a,90)),"peak_p99":float(np.percentile(a,99)),"peak_max":int(a.max()),"pair_product_p90":float(np.percentile(p,90)),"pair_product_p99":float(np.percentile(p,99)),"pair_product_max":int(p.max()),"worst_windows":[{"window_index":int(wi),"n0":int(n0),"n1":int(n1),"product":int(prod)} for prod,wi,n0,n1 in worst[:5]]}
        rows.append(row); (OUT/f"peaks_{name}.json").write_text(json.dumps(row,indent=2),encoding="utf-8")
        del loader,ds,windows,vm; gc.collect(); torch.cuda.empty_cache()
    payload={"status":"passed","threshold_semantics":"training detect_and_match uses raw logit > 0.3","datasets":rows}
    (OUT/"peak_diagnostic.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))

try:
    run()
except Exception as exc:
    payload={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
    (OUT/"fatal_error.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2)); raise
'''
    return prefix+run.replace("__TRAIN__",repr(TRAIN_NAMES)).replace("__VAL__",repr(VAL_NAMES))


def main()->None:
    out=Path(".kaggle_strict_oof_peak_diagnostic_kernel")
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    (out/"run_peak_diagnostic.py").write_text(build_worker(),encoding="utf-8")
    meta={"id":KERNEL_ID,"title":KERNEL_TITLE,"code_file":"run_peak_diagnostic.py","language":"python","kernel_type":"script","is_private":True,"enable_gpu":True,"enable_tpu":False,"enable_internet":False,"keywords":[],"dataset_sources":[SUPPORT_DATASET],"kernel_sources":[],"competition_sources":[COMPETITION],"model_sources":[]}
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps({"kernel":KERNEL_ID,"github_run_id":os.environ.get("GITHUB_RUN_ID","local")},indent=2))

if __name__=="__main__": main()
