#!/bin/bash

set -euo pipefail

# Run from the repo root, AFTER a chunked saliency + attention export
# (calc_saliency_maps: true, saliency_chunked: true, calc_attention_maps: true).
# CPU-only; computes the attention<->saliency Spearman per slide and a cohort
# summary (median rho, bootstrap CI over slides, Wilcoxon signed-rank).

RESULTS_DIR="checkpoints/checkpoint_2026-04-07-04-58-17/test_results/best_model_fold_1"

SALIENCY_DIR="${RESULTS_DIR}/saliency_tile_scores"
ATTENTION_DIR="${RESULTS_DIR}/attention_tile_scores"
OUTPUT_JSON="${RESULTS_DIR}/attention_saliency_spearman.json"

python -m joint_fusion.testing.saliency_attention_corr \
  --saliency-dir "$SALIENCY_DIR" \
  --attention-dir "$ATTENTION_DIR" \
  --output "$OUTPUT_JSON"
