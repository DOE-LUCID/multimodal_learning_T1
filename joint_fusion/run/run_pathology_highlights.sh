#!/bin/bash

set -euo pipefail

# Make sure to call from the repo root.
#
# Emits the pathologist-facing ROI overlays: each slide gets a *_boxes.png with
# an unfilled square drawn around every high-scoring tile, placed at its true
# position on the slide. By default (BOXES_ONLY=true) the heatmap and
# heatmap-overlay are skipped. Hand the *_boxes.png files (in
# <OUTPUT_DIR>/overlays) to the pathologist.

# Choose exactly one input mode:
# 1) Directory mode: point to a folder of .npz tile-score files
TILE_SCORES_DIR="checkpoints/checkpoint_2026-04-07-04-58-17/test_results/best_model_fold_1/saliency_tile_scores"

# 2) Single-file mode: set a specific .npz file and leave TILE_SCORES_DIR empty
TILE_SCORES_FILE=""

# Output folder for generated *_boxes.png ROI overlays (and heatmap/overlay if
# BOXES_ONLY=false).
OUTPUT_DIR="checkpoints/checkpoint_2026-04-07-04-58-17/test_results/best_model_fold_1/pathology_highlights"

# Directory containing full-slide images such as assets/TCGA-50-5044-01Z-00-DX1....png
SLIDE_ASSETS_DIR="assets/png_files_all_samples_per_patient"

# CSV with per-slide full-resolution Width/Height (openslide_getstats.py output).
# REQUIRED for the boxes: they are placed at absolute slide positions, so there
# is no tissue-bbox fallback (a misplaced box is worse than none).
SLIDE_STATS_CSV="assets/slide_statistics_all.csv"

# Tile edge length used when placing each tile on the heatmap canvas
TILE_SIZE=256

# Which scores drive the ROIs. "saliency_scores" for the saliency ROIs, or
# "attention_scores" to box the attention hotspots instead.
SCORE_KEY="saliency_scores"

# ----------------------------------------------------------------------------
# ROI selection (how tiles get boxed)
# ----------------------------------------------------------------------------
# Default: rank-based top (100 - percentile)% of tiles per slide. This gives a
# bounded, consistent number of boxes on EVERY slide (e.g. 90 -> top 10%),
# which is what you want for a comparable pathologist workload.
#   90 -> top 10%   |   95 -> top 5%   |   98 -> top 2%
HIGHLIGHT_PERCENTILE=90

# Optional absolute cutoff on the within-slide min-max-normalized score, in
# [0, 1]. If set (non-empty), it OVERRIDES the percentile. Note this is not
# comparable across slides (each slide's max is 1.0 by construction) and tends
# to be sparse/variable, so prefer the percentile unless you have a reason.
# Example: HIGHLIGHT_THRESHOLD=0.8
HIGHLIGHT_THRESHOLD=""

# Box appearance
BOX_COLOR="lime"
BOX_LINEWIDTH=1.5
# Font size for the per-box rank labels (1, 2, ... matching the exported tile
# file names). Bump up if the numbers are hard to read on dense slides.
BOX_FONTSIZE=6

# Emit ONLY the *_boxes.png ROI overlays (skip the heatmap and heatmap-overlay).
# Set to false to also produce the heatmap + heatmap-overlay alongside the boxes.
BOXES_ONLY=true

# ----------------------------------------------------------------------------
# Pathology-review tile export (the ranked tiles a pathologist opens)
# ----------------------------------------------------------------------------
# When true, also copy the top-scoring tile IMAGES per slide into
# <OUTPUT_DIR>/pathology_review_tiles/<slide_key>/, ranked most-important first.
# Each file is prefixed with its 1-based rank (01_, 02_, ... NN_) so the folder
# browses in score order and the number IS the rank; a tile_scores.csv manifest
# records rank/score/coords. This is independent of the *_boxes.png overlay.
#
# The exported set uses the SAME top-(100 - HIGHLIGHT_PERCENTILE)% selection as
# the boxes, so the number of tiles follows the top 10% per slide (not a fixed
# count) and the exported tiles are exactly the boxed regions of interest.
EXPORT_REVIEW_TILES=true

# Directory holding the ORIGINAL tile images (config.data.input_wsi_path). Tile
# file names come from the .npz; the images are looked up here. REQUIRED when
# EXPORT_REVIEW_TILES=true.
TILES_SOURCE_DIR="/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_20X_1000tiles/tiles/256px_128um/combined/"

CMD=(
  python -m joint_fusion.testing.generate_slide_heatmap_overlays
  --output-dir "$OUTPUT_DIR"
  --slide-assets-dir "$SLIDE_ASSETS_DIR"
  --slide-stats-csv "$SLIDE_STATS_CSV"
  --tile-size "$TILE_SIZE"
  --highlight-boxes
  --box-color "$BOX_COLOR"
  --box-linewidth "$BOX_LINEWIDTH"
  --box-fontsize "$BOX_FONTSIZE"
)

if [[ "$BOXES_ONLY" == "true" ]]; then
  CMD+=(--boxes-only)
fi

if [[ "$EXPORT_REVIEW_TILES" == "true" ]]; then
  CMD+=(--export-review-tiles --tiles-source-dir "$TILES_SOURCE_DIR")
fi

if [[ -n "$TILE_SCORES_FILE" ]]; then
  CMD+=(--tile-scores "$TILE_SCORES_FILE")
elif [[ -n "$TILE_SCORES_DIR" ]]; then
  CMD+=(--tile-scores-dir "$TILE_SCORES_DIR")
else
  echo "Set either TILE_SCORES_FILE or TILE_SCORES_DIR." >&2
  exit 1
fi

if [[ -n "$SCORE_KEY" ]]; then
  CMD+=(--score-key "$SCORE_KEY")
fi

# Absolute threshold overrides the percentile when provided.
if [[ -n "$HIGHLIGHT_THRESHOLD" ]]; then
  CMD+=(--highlight-threshold "$HIGHLIGHT_THRESHOLD")
else
  CMD+=(--highlight-percentile "$HIGHLIGHT_PERCENTILE")
fi

"${CMD[@]}"
