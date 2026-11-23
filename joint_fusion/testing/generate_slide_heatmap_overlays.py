import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from joint_fusion.testing.reconstruct_slide_heatmap import (
    build_heatmap,
    infer_score_key,
    infer_tile_footprint,
    normalize_scores,
)


def args():
    parser = argparse.ArgumentParser(
        description="Generate slide-level heatmaps and overlays from saved tile-score .npz files."
    )
    parser.add_argument(
        "--tile-scores",
        type=str,
        default=None,
        help="Path to a single attention_scores_*.npz or saliency_scores_*.npz file.",
    )
    parser.add_argument(
        "--tile-scores-dir",
        type=str,
        default=None,
        help="Directory containing tile-score .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where heatmaps and overlays will be written.",
    )
    parser.add_argument(
        "--slide-assets-dir",
        type=str,
        default="assets",
        help="Directory containing full-slide images named with the TCGA slide prefix.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Tile edge length in pixels.",
    )
    parser.add_argument(
        "--score-key",
        type=str,
        default=None,
        help="Optional score key override. Defaults to attention_scores or saliency_scores.",
    )
    parser.add_argument(
        "--save-array",
        action="store_true",
        help="Also save the heatmap canvas as a .npy file.",
    )
    parser.add_argument(
        "--slide-stats-csv",
        type=str,
        default=None,
        help="Path to slide_statistics_all.csv (openslide_getstats.py output). Its "
        "per-slide full-resolution Width/Height let the heatmap be placed at its "
        "true absolute position on the thumbnail (no rescaling guess).",
    )
    parser.add_argument(
        "--allow-tissue-fallback",
        action="store_true",
        help="If a slide's dimensions are not found in the stats CSV, fall back to "
        "approximate tissue-bbox registration instead of skipping its overlay.",
    )
    parser.add_argument(
        "--highlight-boxes",
        action="store_true",
        help="Also emit a pathologist-facing overlay that draws a square border "
        "around each high-scoring tile (regions of interest to verify).",
    )
    parser.add_argument(
        "--boxes-only",
        action="store_true",
        help="Emit ONLY the pathologist ROI boxes (implies --highlight-boxes); "
        "skip the heatmap and heatmap-overlay rendering entirely.",
    )
    parser.add_argument(
        "--highlight-percentile",
        type=float,
        default=90.0,
        help="Tiles at/above this within-slide score percentile get a box "
        "(default 90 = top 10%%). Ignored if --highlight-threshold is set.",
    )
    parser.add_argument(
        "--highlight-threshold",
        type=float,
        default=None,
        help="Absolute normalized-score cutoff in [0, 1] for boxing a tile. "
        "Overrides --highlight-percentile when provided.",
    )
    parser.add_argument(
        "--box-color",
        type=str,
        default="lime",
        help="Edge color for highlight boxes.",
    )
    parser.add_argument(
        "--box-linewidth",
        type=float,
        default=1.5,
        help="Edge line width for highlight boxes.",
    )
    parser.add_argument(
        "--box-fontsize",
        type=float,
        default=6.0,
        help="Font size for the per-box rank labels on the highlight overlay.",
    )
    parser.add_argument(
        "--export-review-tiles",
        action="store_true",
        help="For each slide, copy the top-scoring tile images into "
        "<output-dir>/pathology_review_tiles/<patient>/ for pathology review, "
        "with a per-patient tile_scores.csv manifest.",
    )
    parser.add_argument(
        "--tiles-source-dir",
        type=str,
        default=None,
        help="Directory holding the original tile images (config.data.input_wsi_path). "
        "Required with --export-review-tiles; tile file names come from the .npz.",
    )
    parser.add_argument(
        "--review-tiles-dir",
        type=str,
        default=None,
        help="Base directory for the per-patient review folders. "
        "Defaults to <output-dir>/pathology_review_tiles.",
    )
    return parser.parse_args()


def infer_slide_key_from_tile_name(tile_name):
    tile_stem = Path(str(tile_name)).stem
    parts = tile_stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Could not infer slide key from tile name: {tile_name}")
    return "-".join(parts[:3])


def resolve_slide_asset(slide_assets_dir, slide_key):
    assets_dir = Path(slide_assets_dir)
    matches = sorted(assets_dir.glob(f"{slide_key}-*.png"))
    if not matches:
        matches = sorted(assets_dir.glob(f"{slide_key}-*"))
    if not matches:
        return None

    for match in matches:
        if "-DX1" in match.name:
            return match
    return matches[0]


def render_heatmap(heatmap, output_path, title):
    plt.figure(figsize=(12, 12))
    plt.imshow(heatmap, cmap="hot", origin="upper")
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_slide_dimensions(stats_csv_path):
    """Map slide identifiers -> (full_res_width, full_res_height) from the
    openslide_getstats.py CSV.

    Returns two dicts: one keyed by the full file stem (e.g.
    ``TCGA-05-4244-01Z-00-DX1.<uuid>``) for exact matches, and one keyed by the
    3-token slide key (``TCGA-05-4244``) as a fallback, preferring the DX1 slide.
    """
    by_stem = {}
    by_key = {}
    with open(stats_csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = row.get("File", "") or ""
            try:
                width = int(float(row["Width"]))
                height = int(float(row["Height"]))
            except (KeyError, ValueError, TypeError):
                continue

            stem = Path(file_name).stem  # strips .svs
            by_stem[stem] = (width, height)

            parts = stem.split("-")
            if len(parts) >= 3:
                key = "-".join(parts[:3])
                if key not in by_key or "-DX1" in stem:
                    by_key[key] = (width, height)
    return by_stem, by_key


def lookup_slide_dimensions(dims_by_stem, dims_by_key, slide_asset_path, slide_key):
    if dims_by_stem is None:
        return None
    dims = dims_by_stem.get(Path(slide_asset_path).stem)
    if dims is None:
        dims = dims_by_key.get(slide_key)
    return dims


def render_overlay_faithful(
    heatmap, heatmap_extent, slide_asset_path, slide_w, slide_h, output_path, title
):
    """Overlay the heatmap at its true absolute position on the slide.

    The thumbnail is drawn spanning the full-resolution slide coordinate frame
    ``(0, slide_w, slide_h, 0)``, and the heatmap is drawn at its own level-0
    ``heatmap_extent``. Both layers therefore share one coordinate system, so
    the mapping is a single uniform scale (the thumbnail's downsample) with no
    per-axis stretching or tissue-detection heuristic.
    """
    slide_image = plt.imread(slide_asset_path)

    plt.figure(figsize=(12, 12))
    plt.imshow(slide_image, origin="upper", extent=(0, slide_w, slide_h, 0))
    plt.imshow(
        heatmap,
        cmap="hot",
        origin="upper",
        extent=heatmap_extent,
        alpha=0.45,
    )
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def detect_tissue_bbox(slide_image, dark_threshold=150):
    """Bounding box (x0, y0, x1, y1) of tissue in the slide thumbnail.

    Mirrors the tissue test used in preprocessing/remove_background.py: pixels
    darker than ``dark_threshold`` (on a 0-255 scale) count as tissue. Returns
    the full-image box if no tissue is found.
    """
    arr = np.asarray(slide_image)
    if arr.ndim == 3:
        gray = arr[..., :3].mean(axis=2)
    else:
        gray = arr
    # matplotlib returns floats in [0, 1] for PNGs; rescale the threshold then.
    threshold = dark_threshold / 255.0 if gray.max() <= 1.0 else dark_threshold

    mask = gray < threshold
    height, width = gray.shape
    if not mask.any():
        return 0, 0, width, height

    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def render_overlay_tissue(heatmap, slide_asset_path, output_path, title):
    """Approximate fallback when slide dimensions are unavailable.

    Registers the heatmap's bounding box to the thumbnail's detected tissue box.
    This is only an estimate of the slide downsample (it scales x and y
    independently and depends on tissue detection), so it can skew the geometry;
    prefer render_overlay_faithful whenever slide dimensions are known.
    """
    slide_image = plt.imread(slide_asset_path)
    slide_h, slide_w = slide_image.shape[:2]
    x0, y0, x1, y1 = detect_tissue_bbox(slide_image)

    plt.figure(figsize=(12, 12))
    plt.imshow(
        slide_image,
        origin="upper",
        extent=(0, slide_w, slide_h, 0),
    )
    plt.imshow(
        heatmap,
        cmap="hot",
        origin="upper",
        extent=(x0, x1, y1, y0),
        alpha=0.45,
    )
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def select_top_tile_indices(raw_scores, percentile=90.0, threshold=None, count=None):
    """Select the high-scoring tiles for a slide.

    Returns ``(kept_indices, cutoff)`` where ``kept_indices`` are tile indices
    ordered by DESCENDING score and ``cutoff`` is the effective score cutoff.

    Selection priority (first that is set wins):
      * ``count`` (absolute): the top ``count`` tiles by score.
      * ``threshold`` (absolute): scores are min-max normalized within the slide
        (same as the heatmap) and tiles with ``norm >= threshold`` are kept.
      * ``percentile`` (default): the top ``100 - percentile`` fraction of tiles
        by RANK. Rank selection keeps the count bounded (~top 10% at
        ``percentile=90``) even when many tiles are tied at the percentile value.

    Using ``count`` and ``percentile`` interchangeably yields the SAME tile set
    whenever they resolve to the same k (both take the top-k by score), so the
    exported review tiles line up with the boxed tiles on the overlay.
    """
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    n_tiles = raw_scores.size
    order_desc = np.argsort(raw_scores)[::-1]  # highest score first

    if count is not None:
        k = max(1, min(int(count), n_tiles))
        kept = order_desc[:k]
        cutoff = float(raw_scores[kept[-1]])
    elif threshold is not None:
        norm = normalize_scores(raw_scores)
        keep_mask = norm >= float(threshold)
        kept = order_desc[keep_mask[order_desc]]
        cutoff = float(threshold)
    else:
        k = int(np.ceil(n_tiles * (100.0 - percentile) / 100.0))
        k = max(1, min(k, n_tiles))
        kept = order_desc[:k]
        cutoff = float(raw_scores[kept[-1]])

    return kept, cutoff


def compute_highlight_boxes(
    payload, score_key, tile_size, percentile=90.0, threshold=None
):
    """Return the level-0 boxes (x, y, size) for the high-scoring tiles, plus the
    cutoff actually used and the number of boxes.

    Boxes are placed at each tile's true level-0 footprint, matching
    build_heatmap's convention so they line up with the heatmap overlay exactly.
    """
    x_coords = np.asarray(payload["x_coords"], dtype=np.int64)
    y_coords = np.asarray(payload["y_coords"], dtype=np.int64)
    raw_scores = np.asarray(payload[score_key], dtype=np.float64)
    footprint = infer_tile_footprint(x_coords, y_coords, tile_size)

    kept, cutoff = select_top_tile_indices(
        raw_scores, percentile=percentile, threshold=threshold
    )
    boxes = [
        (int(x_coords[i]), int(y_coords[i]), int(footprint)) for i in kept
    ]
    return boxes, cutoff, len(kept)


def render_overlay_boxes(
    boxes, slide_asset_path, slide_w, slide_h, output_path, title,
    box_color="lime", box_linewidth=1.5, box_fontsize=6.0,
):
    """Draw the slide thumbnail with an unfilled square around each high-scoring
    tile, at its true absolute level-0 position (same coordinate frame as
    render_overlay_faithful, so no rescaling guess).

    ``boxes`` are expected in descending-score (rank) order; each box is labelled
    with its 1-based rank so the overlay lines up with the exported review-tile
    file names (``01_``, ``02_``, ...).
    """
    slide_image = plt.imread(slide_asset_path)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(slide_image, origin="upper", extent=(0, slide_w, slide_h, 0))
    for rank, (x, y, size) in enumerate(boxes, start=1):
        ax.add_patch(
            Rectangle(
                (x, y), size, size,
                fill=False, edgecolor=box_color, linewidth=box_linewidth,
            )
        )
        # Rank label sits just above the box's top-left corner (y grows downward).
        ax.text(
            x, y - 0.15 * size, str(rank),
            color=box_color, fontsize=box_fontsize, fontweight="bold",
            ha="left", va="bottom", clip_on=True,
        )
    ax.set_xlim(0, slide_w)
    ax.set_ylim(slide_h, 0)
    ax.set_title(f"{title} - regions of interest (n={len(boxes)}, ranked by score)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def maybe_render_highlight_boxes(
    payload, score_key, tile_size, slide_asset_path, dims, overlay_dir, stem, title,
    percentile, threshold, box_color, box_linewidth, box_fontsize=6.0,
):
    """Emit the pathologist-facing boxed overlay when slide dimensions are known.

    Requires true slide dimensions: the boxes carry no meaning unless placed at
    their absolute position, so (unlike the heatmap) there is no tissue-bbox
    fallback here.
    """
    if dims is None:
        print(
            f"[warn] no slide dimensions for {stem}; skipping highlight boxes "
            f"(needs --slide-stats-csv for absolute placement)"
        )
        return

    boxes, cutoff, n = compute_highlight_boxes(
        payload, score_key, tile_size, percentile=percentile, threshold=threshold
    )
    slide_w, slide_h = dims
    boxes_output = overlay_dir / f"{stem}_boxes.png"
    render_overlay_boxes(
        boxes, slide_asset_path, slide_w, slide_h, boxes_output, title,
        box_color=box_color, box_linewidth=box_linewidth, box_fontsize=box_fontsize,
    )
    print(f"[ok] {stem} -> {boxes_output.name} ({n} tiles boxed, cutoff={cutoff:.4g})")


def export_review_tiles(
    payload, score_key, slide_key, tiles_source_dir, review_root,
    percentile=90.0, threshold=None,
):
    """Copy the top-scoring tile images for one slide into
    ``review_root/<slide_key>/`` and write a tile_scores.csv manifest.

    The exported set is selected with the SAME within-slide percentile/threshold
    as the highlight boxes (default: top 10%), so the tiles a pathologist opens
    are exactly the boxed regions of interest -- the count follows the top-10%%,
    not a fixed number.

    Tile images are looked up by their file name (as stored in the .npz) inside
    ``tiles_source_dir`` (the original tile directory, config.data.input_wsi_path).
    Copied files are prefixed with their 1-based rank so the folder browses in
    score order. Returns the per-patient output directory.
    """
    tile_names = payload.get("tile_names")
    if tile_names is None or len(tile_names) == 0:
        print(f"[warn] {slide_key}: no tile names in payload; skipping tile export")
        return None

    raw_scores = np.asarray(payload[score_key], dtype=np.float64)
    x_coords = np.asarray(payload["x_coords"], dtype=np.int64)
    y_coords = np.asarray(payload["y_coords"], dtype=np.int64)
    norm_scores = normalize_scores(raw_scores)

    kept, cutoff = select_top_tile_indices(
        raw_scores, percentile=percentile, threshold=threshold
    )

    patient_dir = Path(review_root) / slide_key
    patient_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(tiles_source_dir)

    manifest_rows = []
    copied = missing = 0
    for rank, idx in enumerate(kept, start=1):
        tile_name = str(tile_names[idx])
        source_path = source_dir / tile_name
        if not source_path.exists():
            # tolerate a differing image extension for the same tile stem
            candidates = sorted(source_dir.glob(Path(tile_name).stem + ".*"))
            source_path = candidates[0] if candidates else None

        saved_as = None
        if source_path is not None and source_path.exists():
            saved_as = f"{rank:02d}_{Path(tile_name).name}"
            shutil.copy2(source_path, patient_dir / saved_as)
            copied += 1
        else:
            missing += 1
            print(f"[warn] {slide_key}: tile image not found for {tile_name}")

        manifest_rows.append(
            {
                "rank": rank,
                "tile_name": tile_name,
                "saved_as": saved_as,
                "score": float(raw_scores[idx]),
                "normalized_score": float(norm_scores[idx]),
                "x_coord": int(x_coords[idx]),
                "y_coord": int(y_coords[idx]),
            }
        )

    manifest_path = patient_dir / "tile_scores.csv"
    with open(manifest_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank", "tile_name", "saved_as", "score",
                "normalized_score", "x_coord", "y_coord",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        f"[ok] {slide_key} -> {patient_dir} "
        f"({copied} tiles copied, {missing} missing, cutoff={cutoff:.4g})"
    )
    return patient_dir


def iter_tile_score_files(tile_scores, tile_scores_dir):
    if tile_scores is not None:
        yield Path(tile_scores)
        return

    if tile_scores_dir is None:
        raise ValueError("Provide either --tile-scores or --tile-scores-dir.")

    for path in sorted(Path(tile_scores_dir).glob("*.npz")):
        yield path


def process_tile_score_file(
    path, output_dir, slide_assets_dir, tile_size, score_key, save_array,
    dims_by_stem=None, dims_by_key=None, allow_tissue_fallback=False,
    highlight_boxes=False, highlight_percentile=90.0, highlight_threshold=None,
    box_color="lime", box_linewidth=1.5, box_fontsize=6.0, boxes_only=False,
    export_review_tiles_flag=False, tiles_source_dir=None,
    review_tiles_root=None,
):
    payload = np.load(path, allow_pickle=True)
    resolved_score_key = infer_score_key(payload, score_key)
    title = resolved_score_key.replace("_", " ")

    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # boxes_only implies the pathologist ROI boxes; the heatmap/overlay are skipped.
    draw_boxes = highlight_boxes or boxes_only

    # Slide asset + dimensions are needed by both the overlay and the boxes.
    tile_names = payload.get("tile_names")
    if tile_names is None or len(tile_names) == 0:
        return

    slide_key = infer_slide_key_from_tile_name(tile_names[0])

    # Pathology-review tile export is independent of the slide thumbnail asset:
    # it copies the top tiles straight from the source tile directory.
    if export_review_tiles_flag:
        export_review_tiles(
            payload=payload,
            score_key=resolved_score_key,
            slide_key=slide_key,
            tiles_source_dir=tiles_source_dir,
            review_root=review_tiles_root,
            percentile=highlight_percentile,
            threshold=highlight_threshold,
        )

    slide_asset_path = resolve_slide_asset(slide_assets_dir, slide_key)
    if slide_asset_path is None:
        print(f"[warn] no slide asset found for {slide_key}")
        return

    dims = lookup_slide_dimensions(
        dims_by_stem, dims_by_key, slide_asset_path, slide_key
    )

    if not boxes_only:
        heatmap, heatmap_extent = build_heatmap(payload, resolved_score_key, tile_size)

        heatmap_dir = output_dir / "heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        heatmap_output = heatmap_dir / f"{path.stem}_heatmap.png"
        render_heatmap(heatmap, heatmap_output, title)

        if save_array:
            np.save(heatmap_output.with_suffix(".npy"), heatmap)

        overlay_output = overlay_dir / f"{path.stem}_overlay.png"
        if dims is not None:
            slide_w, slide_h = dims
            render_overlay_faithful(
                heatmap,
                heatmap_extent,
                slide_asset_path,
                slide_w,
                slide_h,
                overlay_output,
                title,
            )
            mode = "faithful"
        elif allow_tissue_fallback:
            render_overlay_tissue(heatmap, slide_asset_path, overlay_output, title)
            mode = "tissue-fallback"
        else:
            print(
                f"[warn] no slide dimensions for {slide_key} in stats CSV; "
                f"skipping overlay (use --allow-tissue-fallback to approximate)"
            )
            # No overlay, but boxes may still be requested below.
            mode = None

        if mode is not None:
            print(
                f"[ok] {path.name} -> {heatmap_output.name}, "
                f"{overlay_output.name} ({mode})"
            )

    if draw_boxes:
        maybe_render_highlight_boxes(
            payload=payload,
            score_key=resolved_score_key,
            tile_size=tile_size,
            slide_asset_path=slide_asset_path,
            dims=dims,
            overlay_dir=overlay_dir,
            stem=path.stem,
            title=title,
            percentile=highlight_percentile,
            threshold=highlight_threshold,
            box_color=box_color,
            box_linewidth=box_linewidth,
            box_fontsize=box_fontsize,
        )


def main():
    opt = args()
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dims_by_stem = dims_by_key = None
    if opt.slide_stats_csv:
        dims_by_stem, dims_by_key = load_slide_dimensions(opt.slide_stats_csv)
        print(
            f"[info] loaded full-resolution dimensions for {len(dims_by_stem)} "
            f"slides from {opt.slide_stats_csv}"
        )
    else:
        print(
            "[warn] no --slide-stats-csv provided; overlays need slide dimensions "
            "for faithful placement (use --allow-tissue-fallback to approximate)"
        )

    review_tiles_root = None
    if opt.export_review_tiles:
        if not opt.tiles_source_dir:
            raise ValueError(
                "--export-review-tiles requires --tiles-source-dir (the original "
                "tile directory, i.e. config.data.input_wsi_path)."
            )
        review_tiles_root = Path(opt.review_tiles_dir or (output_dir / "pathology_review_tiles"))
        review_tiles_root.mkdir(parents=True, exist_ok=True)
        selection = (
            f"score >= {opt.highlight_threshold}"
            if opt.highlight_threshold is not None
            else f"top {100.0 - opt.highlight_percentile:g}%"
        )
        print(
            f"[info] exporting review tiles per slide ({selection}, same set as the "
            f"highlight boxes) into {review_tiles_root}"
        )

    for path in iter_tile_score_files(opt.tile_scores, opt.tile_scores_dir):
        process_tile_score_file(
            path=path,
            output_dir=output_dir,
            slide_assets_dir=opt.slide_assets_dir,
            tile_size=opt.tile_size,
            score_key=opt.score_key,
            save_array=opt.save_array,
            dims_by_stem=dims_by_stem,
            dims_by_key=dims_by_key,
            allow_tissue_fallback=opt.allow_tissue_fallback,
            highlight_boxes=opt.highlight_boxes,
            highlight_percentile=opt.highlight_percentile,
            highlight_threshold=opt.highlight_threshold,
            box_color=opt.box_color,
            box_linewidth=opt.box_linewidth,
            box_fontsize=opt.box_fontsize,
            boxes_only=opt.boxes_only,
            export_review_tiles_flag=opt.export_review_tiles,
            tiles_source_dir=opt.tiles_source_dir,
            review_tiles_root=review_tiles_root,
        )


if __name__ == "__main__":
    main()
