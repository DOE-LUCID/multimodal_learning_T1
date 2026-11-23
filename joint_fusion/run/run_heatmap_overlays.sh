#!/bin/bash

set -euo pipefail

# Make sure to call from the repo root.

# Choose exactly one input mode:
# 1) Directory mode: point to a folder of .npz tile-score files
TILE_SCORES_DIR="checkpoints/checkpoint_2026-04-07-04-58-17/test_results/best_model_fold_1/attention_tile_scores"

# 2) Single-file mode: set a specific .npz file and leave TILE_SCORES_DIR empty
TILE_SCORES_FILE=""

# Output folder for generated heatmaps and overlays
OUTPUT_DIR="checkpoints/checkpoint_2026-04-07-04-58-17/test_results/best_model_fold_1/attention_visualizations"

# Directory containing full-slide images such as assets/TCGA-50-5044-01Z-00-DX1....png
SLIDE_ASSETS_DIR="assets/png_files_all_samples_per_patient"

# CSV with per-slide full-resolution Width/Height (openslide_getstats.py output).
# Enables faithful (no-skew) placement of the heatmap on the thumbnail.
SLIDE_STATS_CSV="assets/slide_statistics_all.csv"

# Tile edge length used when placing each tile on the heatmap canvas
TILE_SIZE=256

# Optional: set to "attention_scores" or "saliency_scores" if you want to force the key
# Leave empty to auto-detect from the .npz file
SCORE_KEY=""

# Optional: set to true to also save the heatmap canvas as .npy
SAVE_ARRAY=false

CMD=(
  python -m joint_fusion.testing.generate_slide_heatmap_overlays
  --output-dir "$OUTPUT_DIR"
  --slide-assets-dir "$SLIDE_ASSETS_DIR"
  --slide-stats-csv "$SLIDE_STATS_CSV"
  --tile-size "$TILE_SIZE"
)

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

if [[ "$SAVE_ARRAY" == "true" ]]; then
  CMD+=(--save-array)
fi

"${CMD[@]}"
