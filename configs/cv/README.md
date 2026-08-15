# Biohub CV policy

## `balanced5-v1`

The project uses a deterministic 5-fold dataset-level split derived from the
199 training GEFF graphs.  This is the same algorithm used by the successful
`biohub_cv_audit` v7 run.

1. Sort all dataset names lexicographically when building the source manifest.
2. Sort manifest rows by `(gt_edges, gt_divisions, gt_nodes)` descending.
3. Maintain five initially empty folds with cumulative `edges`, `divisions`,
   and `nodes` counters.
4. For each dataset in that order, assign it to the fold minimizing
   `(edges, divisions, nodes, fold_id)` lexicographically.
5. `valid` is the assigned dataset list; `train` is the complement over all
   199 manifest datasets.

The v7 audit produced validation sizes 40/40/40/40/39, with validation edge
counts 25782/25785/25790/25785/25741 and division counts 35/32/29/22/33.

### Rules

- Split at the **dataset/video level**, never by frame or temporal window.
- No training, pseudo-label fitting, threshold fitting, or graph calibration may
  use the validation GEFF/image of the fold being scored unless the experiment
  is explicitly labeled non-OOF.
- Public pretrained weights with unknown training membership must be labeled
  `non_oof_pretrained_reference_unknown_train_overlap`; they are diagnostics,
  not CV evidence.
- Report the official micro score together with per-dataset results and the
  detection/linking decomposition (`edge_endpoint_coverage` and
  `conditional_link_recall`).
- Keep this policy fixed for model comparison. A changed split definition gets
  a new version name rather than silently replacing `balanced5-v1`.
