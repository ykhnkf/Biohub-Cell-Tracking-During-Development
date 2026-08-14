# Biohub OOF Validation Framework

## Goal

Build a validation system that is independent of Public LB and can distinguish detection failures from linking failures.

## Split policy

### Primary: dataset-level OOF

Never split frames from one dataset/embryo across train and validation.

Preferred split after the Phase-0 audit:

- **LODO (leave-one-dataset-out)** when the number of independent datasets is modest.
- **Balanced 5-fold Group CV** when enough independent datasets exist. Balance by GT edge count first, then division count and node count.

### Secondary: temporal stress test

Within a held-out dataset, report early/middle/late time blocks as diagnostics only. Temporal splits are not the primary CV because the same embryo remains in both train and validation.

## Detection diagnostics

For each OOF dataset report:

1. `annotated_node_recall`: fraction of GT nodes matched by a prediction within the official 7 µm matching rule.
2. `localization_distance_um`: median / p90 physical distance for matched nodes.
3. `predicted_node_count`: total and per-frame node count.
4. `pred_to_estimated_node_ratio`: predicted node count relative to GT metadata estimate when available.
5. `edge_endpoint_coverage`: fraction of GT edges for which both endpoints have matched predicted nodes.

`edge_endpoint_coverage` is the practical upper bound on ordinary edge recall for the current detector.

## Linking diagnostics

Evaluate linking in two modes.

### A. Oracle-node linking

Feed GT nodes (optionally with realistic localization noise) into the linker. This isolates association quality from detector failures.

Report:

- GT edge recall
- wrong-edge FP
- conditional linking recall
- division edge recall

Do not use adjusted edge Jaccard as the primary oracle-node metric because GT is sparse and its node count does not represent the full estimated cell population.

### B. Predicted-node end-to-end

Run the exact submission graph pipeline on OOF predictions:

`detector -> candidate nodes -> linker -> graph optimizer -> gap recovery -> track filtering -> division recovery`

Then run the official evaluator.

This is the primary model-selection score.

## Detection / linking decomposition

For every GT edge classify it as:

1. **missing-node**: at least one endpoint was not detected;
2. **recoverable-but-unlinked**: both endpoints were detected, but the correct edge was absent;
3. **correctly-linked**: both endpoints detected and correct edge predicted.

Define:

- `detection_edge_coverage = recoverable_gt_edges / all_gt_edges`
- `conditional_link_recall = correct_edges / recoverable_gt_edges`

This lets us determine whether the next unit of effort belongs in detection or linking.

## Official end-to-end metrics

For every held-out dataset retain raw counts, not just scores:

- edge TP / FP / FN
- division TP / FP / FN
- number of predicted nodes
- node recall
- raw edge Jaccard
- adjusted edge Jaccard

Aggregate the full OOF set with the official summarisation routine. Do not average fold scores arithmetically.

## Model-selection policy

A change is considered robust only when:

1. aggregate OOF adjusted-edge/combined score improves;
2. the gain is not driven by only one held-out dataset;
3. worst-dataset performance does not collapse without a justified trade-off;
4. detection/linking decomposition identifies a plausible mechanism for the gain.

Public LB is a low-frequency external check, not a hyperparameter tuner.

## Phase sequence

### Phase 0 — audit

- inspect all training GT datasets
- confirm official metric self-test
- generate LODO and balanced group folds

### Phase 1 — public baseline OOF

- reproduce public detector/linker on each fold
- save OOF nodes and graph predictions
- compute detection/linking decomposition

### Phase 2 — cheap ablations

- detection threshold / NMS radius
- edge threshold / motion parameters
- gap recovery / short-track filtering

Use cached OOF detector outputs whenever possible so graph-level sweeps do not retrain the 3D model.

### Phase 3 — model changes

Only after the decomposition shows the bottleneck:

- detector improvements if edge endpoint coverage is limiting;
- linker improvements if conditional link recall is limiting.
