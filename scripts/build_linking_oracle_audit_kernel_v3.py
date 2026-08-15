from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from build_cv_audit_kernel_v3 import COMPETITION, SUPPORT_DATASET
from build_linking_oracle_audit_kernel import KERNEL_TITLE, build_worker as build_worker_v1

KERNEL_ID = "ykhnkf/biohub-gt-node-linking-oracle-audit"


def build_worker() -> str:
    worker = build_worker_v1()
    old = '''    def make_pred(attrs, scale, gate):
        g = td.graph.InMemoryGraph()
        # The Kaggle support pack currently installs a TracksData/Polars
        # combination where the Polars DataTypeClass (pl.Float64) is rejected
        # by pl.Series(dtype=...).  Passing the instantiated dtype is accepted
        # by Polars while preserving a Float64 TracksData schema.
        for key in ["z", "y", "x"]:
            g.add_node_attr_key(key, dtype=pl.Float64(), default_value=-999999.0)
        rows = attrs.to_dicts()
        new_ids = g.bulk_add_nodes([{"t":int(r["t"]),"z":float(r["z"]),"y":float(r["y"]),"x":float(r["x"])} for r in rows])
'''
    new = '''    def make_pred(gt_path, attrs, scale, gate):
        # Clone the GT graph so node IDs, coordinates and node schemas are
        # preserved exactly, then remove every true edge before reconstructing
        # associations.  This avoids creating a new TracksData attribute schema,
        # which is incompatible with the Polars version bundled in this Kaggle
        # image.  The diagnostic intentionally supplies oracle GT nodes; only
        # temporal associations are being tested.
        g = load_graph(gt_path)
        existing_edge_ids = list(g.edge_ids())
        if existing_edge_ids:
            g.bulk_remove_edges(existing_edge_ids)
        rows = attrs.to_dicts()
'''
    if old not in worker:
        raise RuntimeError("Expected make_pred schema block not found")
    worker = worker.replace(old, new, 1)
    worker = worker.replace(
        '''                    edge_rows.append({"source_id":new_ids[ia[rix]],"target_id":new_ids[ib[cix]]})''',
        '''                    edge_rows.append({"source_id":int(rows[ia[rix]]["node_id"]),"target_id":int(rows[ib[cix]]["node_id"])})''',
        1,
    )
    worker = worker.replace(
        '''            pred = make_pred(attrs, scale, gate)''',
        '''            pred = make_pred(path, attrs, scale, gate)''',
        1,
    )
    return worker


def main() -> None:
    out = Path(".kaggle_linking_oracle_kernel")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "run_linking_oracle.py").write_text(build_worker(), encoding="utf-8")
    meta = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "run_linking_oracle.py",
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
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"kernel": KERNEL_ID, "github_run_id": os.environ.get("GITHUB_RUN_ID", "local")}, indent=2))


if __name__ == "__main__":
    main()
