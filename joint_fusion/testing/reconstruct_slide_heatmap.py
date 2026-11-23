import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from joint_fusion.config.config_manager import ConfigManager


def args():
    parser = argparse.ArgumentParser(
        description="Render a slide-level heatmap from tile attribution metadata."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="joint_fusion/config/base_config.yaml",
        help="Config path. Heatmap settings are read from testing.* unless overridden here.",
    )
    parser.add_argument(
        "--tile-scores",
        type=str,
        default=None,
        help="Path to an attention_scores_*.npz or saliency_scores_*.npz file.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tile edge length in pixels used when placing tiles on the slide canvas.",
    )
    parser.add_argument(
        "--score-key",
        type=str,
        default=None,
        help="Optional override for the score array key. Defaults to attention_scores or saliency_scores.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output PNG path. Defaults next to the input file.",
    )
    parser.add_argument(
        "--save-array",
        action="store_true",
        help="Also save the rendered heatmap canvas as a .npy array.",
    )
    return parser.parse_args()


def infer_score_key(payload, requested_key=None):
    if requested_key is not None:
        if requested_key not in payload:
            raise KeyError(f"Score key '{requested_key}' not found in payload.")
        return requested_key

    for key in ("attention_scores", "saliency_scores"):
        if key in payload:
            return key

    raise KeyError("Could not infer score key. Expected attention_scores or saliency_scores.")


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float32)
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    if score_max <= score_min:
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - score_min) / (score_max - score_min)


def infer_tile_footprint(x_coords, y_coords, fallback):
    """Recover the tile footprint (grid stride) in slide-level-0 pixels.

    The tile x/y values are slideflow tile *centers* in full-resolution
    (level-0) pixels, laid out on a regular grid whose spacing equals the
    tile's level-0 footprint (``tile_um / mpp``). The greatest common divisor
    of the gaps between distinct coordinates recovers that stride even when
    only a random subset of tiles was kept. ``fallback`` (the legacy
    ``tile_size``) is used when the grid is too degenerate to infer.
    """
    gaps = []
    for coords in (x_coords, y_coords):
        unique = np.unique(np.asarray(coords, dtype=np.int64))
        if unique.size >= 2:
            diffs = np.diff(unique)
            gaps.append(diffs[diffs > 0])

    if not gaps:
        return int(fallback)

    all_gaps = np.concatenate(gaps)
    if all_gaps.size == 0:
        return int(fallback)

    stride = int(np.gcd.reduce(all_gaps))
    # Guard against a degenerate gcd (e.g. 1) from off-grid coords produced by
    # slide alignment: fall back to the smallest observed gap, then to legacy.
    if stride < max(8, int(fallback) // 4):
        stride = int(all_gaps.min())
    if stride <= 0:
        return int(fallback)
    return stride


def build_heatmap(payload, score_key, tile_size, footprint=None, max_canvas_side=4096):
    x_coords = np.asarray(payload["x_coords"], dtype=np.int64)
    y_coords = np.asarray(payload["y_coords"], dtype=np.int64)
    scores = normalize_scores(payload[score_key])

    # x/y are tile centers in level-0 px; paint each tile at its true level-0
    # footprint so tiles tessellate instead of leaving gaps (or overlapping)
    # when the slide's native magnification differs from the saved tile_px.
    if footprint is None:
        footprint = infer_tile_footprint(x_coords, y_coords, tile_size)

    min_x = int(x_coords.min())
    min_y = int(y_coords.min())

    # The canvas spans the tile bbox in level-0 px, which can be tens of
    # thousands wide. Downsample so the raster stays bounded (the overlay
    # rescales it to the thumbnail anyway, so extra resolution is wasted).
    raw_side = max(
        int(x_coords.max()) - min_x + footprint,
        int(y_coords.max()) - min_y + footprint,
    )
    scale = max(1, -(-raw_side // int(max_canvas_side)))  # ceil division
    footprint_ds = max(1, footprint // scale)

    shifted_x = (x_coords - min_x) // scale
    shifted_y = (y_coords - min_y) // scale

    width = int(shifted_x.max()) + footprint_ds
    height = int(shifted_y.max()) + footprint_ds

    heatmap = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    for x, y, score in zip(shifted_x, shifted_y, scores):
        heatmap[y : y + footprint_ds, x : x + footprint_ds] += score
        counts[y : y + footprint_ds, x : x + footprint_ds] += 1.0

    nonzero = counts > 0
    heatmap[nonzero] /= counts[nonzero]

    # The level-0 (full-resolution) slide region this canvas covers, in
    # matplotlib extent order (left, right, bottom, top) with y increasing
    # downward. This lets the overlay place the heatmap at its true absolute
    # position on the slide, with no rescaling guess.
    extent = (
        min_x,
        min_x + width * scale,
        min_y + height * scale,
        min_y,
    )
    return heatmap, extent


def render_heatmap(heatmap, output_path, title):
    plt.figure(figsize=(12, 12))
    plt.imshow(heatmap, cmap="hot", origin="upper")
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_payload_from_metadata(tile_names, x_coords, y_coords, scores, score_key):
    return {
        "tile_names": np.asarray(tile_names, dtype=object),
        "x_coords": np.asarray(x_coords, dtype=np.int32),
        "y_coords": np.asarray(y_coords, dtype=np.int32),
        score_key: np.asarray(scores, dtype=np.float32),
    }


def render_payload_heatmap(
    payload,
    score_key,
    tile_size,
    output_path,
    title=None,
    save_array=False,
):
    heatmap, _ = build_heatmap(payload, score_key, tile_size)
    resolved_title = title or score_key.replace("_", " ")
    render_heatmap(heatmap, output_path, resolved_title)

    if save_array:
        np.save(Path(output_path).with_suffix(".npy"), heatmap)

    return heatmap


def main():
    opt = args()
    config = ConfigManager.load_config(opt.config)

    tile_scores_value = opt.tile_scores or config.testing.heatmap_tile_scores_path
    if tile_scores_value in (None, ""):
        raise ValueError(
            "No tile score file provided. Set testing.heatmap_tile_scores_path in the config or pass --tile-scores."
        )

    tile_scores_path = Path(tile_scores_value)
    payload = np.load(tile_scores_path, allow_pickle=True)
    score_key_override = opt.score_key or config.testing.heatmap_score_key or None
    score_key = infer_score_key(payload, score_key_override)

    tile_size = opt.tile_size or config.testing.heatmap_tile_size
    output_override = opt.output or config.testing.heatmap_output_path
    if output_override in (None, ""):
        output_path = tile_scores_path.with_name(f"{tile_scores_path.stem}_heatmap.png")
    else:
        output_path = Path(output_override)

    render_payload_heatmap(
        payload=payload,
        score_key=score_key,
        tile_size=tile_size,
        output_path=output_path,
        title=score_key.replace("_", " "),
        save_array=opt.save_array or config.testing.heatmap_save_array,
    )


if __name__ == "__main__":
    main()
