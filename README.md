# Biohub - Cell Tracking During Development

Repository for the Kaggle competition **Biohub - Cell Tracking During Development**.

## Current priority

Build a leaderboard-independent validation system before model iteration.

The validation stack will separate:

1. **Dataset-level generalization** — no frames from the same dataset/embryo may cross train/validation folds.
2. **Detection quality** — annotated-node recall, node-count calibration, and GT-edge endpoint coverage.
3. **Linking quality** — conditional edge recall when both GT endpoints are detectable, wrong-link FP, and oracle-node linking diagnostics.
4. **End-to-end score** — the official adjusted edge Jaccard + 0.1 × division Jaccard, aggregated across OOF datasets.

## Phase 0: dataset/CV audit

Run the GitHub Action **Biohub Kaggle CV Audit**. It launches a private Kaggle kernel against the competition data and returns:

- `dataset_manifest.csv`
- `folds_lodo.json`
- `folds_balanced_5.json`
- `official_metric_selftest.json`
- `summary.json`

The audit intentionally does not train a model. Its job is to inspect the real training-set structure and verify the official metric in the same Kaggle environment we will use for experiments.

## Required GitHub secret

This repository needs a repository secret named:

`KAGGLE_API_TOKEN`

The token is used only by GitHub Actions to push/poll the private Kaggle evaluation kernel and download its outputs.

## Competition slug

`biohub-cell-tracking-during-development`
