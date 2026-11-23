"""Attention<->saliency Spearman analysis (post-hoc, CPU-only).

Reads the per-tile .npz files written by the chunked full-context saliency run
(saliency_tile_scores/) and the attention export (attention_tile_scores/),
aligns them per slide by tile name, and computes:

  - per-slide observed Spearman rho over the slide's tiles, and
  - a cohort summary whose uncertainty comes from a TWO-STAGE (cluster)
    bootstrap that propagates both sampling levels.

Two levels of sampling are present:
  (1) tiles within a slide -- the ~400 tiles are a random subsample of the WSI's
      tissue tiles (slideflow max_tiles + random.sample in preprocessing), so a
      slide's rho is an ESTIMATE, not a population value; and
  (2) slides within the cohort -- the patients are a sample of the population.

The two-stage bootstrap resamples patients (the independent unit) and, within
each resampled patient, resamples its tiles, recomputing the slide rho each time.
The 2.5/97.5 percentiles of the cohort median over many resamples give a 95% CI
that reflects "different patients AND different tiles."

Inference note. The per-slide analytic p-value is dropped on purpose: tiles
within a slide are spatially autocorrelated, so it would be anti-conservative.
The cohort claim rests on the across-patient bootstrap CI and a Wilcoxon
signed-rank test on the per-slide rho values.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe (no display on the remote)
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, wilcoxon

SAL_RE = re.compile(r"saliency_scores_(.+)\.npz$")


def iter_paired_files(saliency_dir, attention_dir):
    """Yield (tcga_id, saliency_path, attention_path) for slides present in both."""
    saliency_dir = Path(saliency_dir)
    attention_dir = Path(attention_dir)
    for sal_path in sorted(saliency_dir.glob("saliency_scores_*.npz")):
        m = SAL_RE.search(sal_path.name)
        if not m:
            continue
        tcga_id = m.group(1)
        att_path = attention_dir / f"attention_scores_{tcga_id}.npz"
        if att_path.exists():
            yield tcga_id, sal_path, att_path


def fast_spearman(a, b):
    """Spearman rho = Pearson on average-ranked data (ties handled). NaN if degenerate.

    Resampling tiles with replacement creates duplicate values; rankdata's
    average-rank method gives them tied ranks, so this is the textbook Spearman
    recomputed on the resampled dataset (not a fixed-rank shortcut).
    """
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    ra = rankdata(a)
    rb = rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    if denom == 0:
        return np.nan
    return float((ra @ rb) / denom)


def load_slide_pairs(saliency_dir, attention_dir):
    """For each slide: aligned (attention, saliency) arrays + observed Spearman rho."""
    slides = []
    for tcga_id, sal_path, att_path in iter_paired_files(saliency_dir, attention_dir):
        sal = np.load(sal_path, allow_pickle=True)
        att = np.load(att_path, allow_pickle=True)
        att_lookup = {
            str(name): float(score)
            for name, score in zip(att["tile_names"], att["attention_scores"])
        }
        paired_att, paired_sal = [], []
        for name, score in zip(sal["tile_names"], sal["saliency_scores"]):
            key = str(name)
            if key in att_lookup:
                paired_att.append(att_lookup[key])
                paired_sal.append(float(score))
        a = np.asarray(paired_att, dtype=np.float64)
        s = np.asarray(paired_sal, dtype=np.float64)
        slides.append(
            {
                "tcga_id": tcga_id,
                "att": a,
                "sal": s,
                "rho": fast_spearman(a, s),
                "n_tiles": int(a.size),
            }
        )
    return slides


def two_stage_bootstrap(slides, n_boot=2000, seed=40, statistic=np.nanmedian):
    """Cluster (two-stage) bootstrap replicates of the cohort statistic.

    Stage 1 -- patients: draw ``n_slides`` slides WITH replacement (independent unit).
    Stage 2 -- tiles: within each drawn slide, draw its tiles WITH replacement and
               recompute that slide's Spearman rho.
    Record ``statistic`` (default median) over the resampled slides for each of the
    ``n_boot`` resamples. Returns the full array of replicate statistics (the
    bootstrap distribution); the caller takes percentiles for the CI and plots it.
    Propagates BOTH sampling levels. Returns None if fewer than 2 usable slides.
    """
    usable = [s for s in slides if not np.isnan(s["rho"])]
    atts = [s["att"] for s in usable]
    sals = [s["sal"] for s in usable]
    n_slides = len(usable)
    if n_slides < 2:
        return None

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        slide_idx = rng.integers(0, n_slides, n_slides)          # stage 1: patients
        rhos = np.empty(n_slides)
        for j, si in enumerate(slide_idx):
            a = atts[si]
            s = sals[si]
            tile_idx = rng.integers(0, a.size, a.size)            # stage 2: tiles
            rhos[j] = fast_spearman(a[tile_idx], s[tile_idx])
        boot_stats[b] = statistic(rhos)

    return boot_stats


def plot_bootstrap_distribution(boot_stats, output_path, median_rho=None, ci=None):
    """Boxplot of the two-stage bootstrap distribution of the cohort median rho,
    in the same format as bootstrap.py's C-index boxplot."""
    boot = np.asarray(boot_stats, dtype=np.float64)
    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        return None

    plt.figure(figsize=(10, 6))
    plt.boxplot(boot, vert=True)
    plt.title("Bootstrapped Attention-Saliency Spearman Correlation")
    plt.xticks([1], ["Multimodal (joint fusion)"])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(output_path)


def summarize_cohort(slides, n_boot=2000, seed=40):
    """Returns (summary_dict, boot_stats) where boot_stats is the bootstrap
    distribution of the cohort median rho (or None)."""
    rhos = np.array([s["rho"] for s in slides], dtype=np.float64)
    valid = rhos[~np.isnan(rhos)]
    summary = {"n_slides_used": int(valid.size)}
    if valid.size == 0:
        return summary, None

    summary.update(
        {
            "median_rho": float(np.median(valid)),
            "mean_rho": float(np.mean(valid)),
            "iqr": [float(np.percentile(valid, 25)), float(np.percentile(valid, 75))],
        }
    )
    boot_stats = two_stage_bootstrap(slides, n_boot=n_boot, seed=seed)
    if boot_stats is not None:
        ci = (
            float(np.nanpercentile(boot_stats, 2.5)),
            float(np.nanpercentile(boot_stats, 97.5)),
        )
        summary["median_two_stage_bootstrap_ci95"] = list(ci)
        summary["bootstrap_n_resamples"] = int(n_boot)

    # Wilcoxon signed-rank: is the per-slide rho distribution centered above 0?
    nonzero = valid[valid != 0]
    if nonzero.size >= 1:
        try:
            stat, p = wilcoxon(nonzero, alternative="greater")
            summary["wilcoxon_stat"] = float(stat)
            summary["wilcoxon_p_greater_than_0"] = float(p)
        except ValueError:
            summary["wilcoxon_p_greater_than_0"] = None
    return summary, boot_stats


def args():
    parser = argparse.ArgumentParser(
        description="Spearman attention<->saliency analysis (two-stage bootstrap) "
        "over saved tile-score .npz files."
    )
    parser.add_argument("--saliency-dir", required=True,
                        help="Directory of saliency_scores_*.npz (saliency_tile_scores).")
    parser.add_argument("--attention-dir", required=True,
                        help="Directory of attention_scores_*.npz (attention_tile_scores).")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--plot", default=None,
                        help="Output PNG path for the bootstrap distribution plot. "
                        "Defaults to <output>_bootstrap.png next to the JSON.")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Number of two-stage bootstrap resamples.")
    parser.add_argument("--bootstrap-seed", type=int, default=40)
    return parser.parse_args()


def main():
    opt = args()
    slides = load_slide_pairs(opt.saliency_dir, opt.attention_dir)

    per_slide = [
        {"tcga_id": s["tcga_id"], "spearman_rho": s["rho"], "n_tiles": s["n_tiles"]}
        for s in slides
    ]
    cohort, boot_stats = summarize_cohort(
        slides, n_boot=opt.bootstrap, seed=opt.bootstrap_seed
    )
    result = {
        "n_slides": len(slides),
        "cohort": cohort,
        "per_slide": per_slide,
    }

    out_path = Path(opt.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    c = result["cohort"]
    print(f"[ok] {result['n_slides']} slides -> {out_path}")
    if "median_rho" in c:
        ci = c.get("median_two_stage_bootstrap_ci95", ["?", "?"])
        print(
            f"     median rho={c['median_rho']:.3f}  "
            f"two-stage 95% CI=[{ci[0]:.3f}, {ci[1]:.3f}]  "
            f"Wilcoxon p(>0)={c.get('wilcoxon_p_greater_than_0')}"
        )

    if boot_stats is not None:
        plot_path = (
            Path(opt.plot)
            if opt.plot
            else out_path.with_name(out_path.stem + "_bootstrap.png")
        )
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_bootstrap_distribution(
            boot_stats,
            plot_path,
            median_rho=c.get("median_rho"),
            ci=c.get("median_two_stage_bootstrap_ci95"),
        )
        print(f"     bootstrap distribution plot -> {plot_path}")


if __name__ == "__main__":
    main()
