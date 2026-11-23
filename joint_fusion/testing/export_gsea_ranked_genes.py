"""
export_gsea_ranked_genes.py
===========================
Write the preranked gene list for fgsea (R) as gsea_ranked_genes.csv, keyed by Ensembl ID with version suffixes removed so the IDs match msigdbr's ``ensembl_gene`` 

Source: interpret_omics `all_gene_ig_scores.csv.` The ranking statistic is ``mean_path_gradient`` (cohort-mean path gradient).

Output columns:
    ensembl_id          Ensembl gene ID, version suffix stripped (ENSG...123.4 -> ENSG...123)
    mean_path_gradient  ranking statistic, sorted descending

In R:
    library(msigdbr); library(fgsea)
    df =  read.csv("gsea_ranked_genes.csv")
    ranks = sort(setNames(df$mean_path_gradient, df$ensembl_id), decreasing = TRUE)
    m =  msigdbr(species = "Homo sapiens", collection = "C6")
    pathways = split(m$ensembl_gene, m$gs_name)     # Ensembl-keyed gene sets
    fgsea(pathways, ranks, minSize = 15, maxSize = 500, eps = 0)

Usage:
    # resolve all_gene_ig_scores.csv from the config's output_base_dir
    python -m joint_fusion.testing.export_gsea_ranked_genes

    # or point straight at the input / output
    python -m joint_fusion.testing.export_gsea_ranked_genes \
        --in <base>/interpret_omics/all_gene_ig_scores.csv --out gsea_ranked_genes.csv
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

DEFAULT_CONFIG = "joint_fusion/config/config_checkpoint_2026-04-07-04-58-17_fold1.yaml"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RANK_COLUMN = "mean_path_gradient"


def strip_ensembl_version(series):
    """ENSG00000141510.17 -> ENSG00000141510 (msigdbr's ensembl_gene is unversioned)."""
    return series.astype(str).str.replace(r"\.\d+$", "", regex=True)


def export_ranked_genes(all_genes_csv, out_path):
    """Read all_gene_ig_scores.csv and write the Ensembl-keyed, version-stripped ranked CSV."""
    df = pd.read_csv(all_genes_csv)
    for col in ("ensembl_id", RANK_COLUMN):
        if col not in df.columns:
            raise ValueError(
                f"{all_genes_csv} has no '{col}' column; expected the interpret_omics "
                "all_gene_ig_scores.csv."
            )

    ranked = df[["ensembl_id", RANK_COLUMN]].copy()
    ranked = ranked.dropna(subset=[RANK_COLUMN])
    if ranked.empty:
        raise ValueError(
            f"No finite {RANK_COLUMN} in {all_genes_csv}; path gradients were not "
            "available when interpret_omics ran, so there is no ranking metric."
        )

    ranked["ensembl_id"] = strip_ensembl_version(ranked["ensembl_id"])

    # Version stripping can collapse two IDs onto one base ID; average them so the
    # ranks vector fgsea receives has unique names.
    n_before = len(ranked)
    ranked = ranked.groupby("ensembl_id", as_index=False)[RANK_COLUMN].mean()
    if len(ranked) < n_before:
        logger.warning(
            f"{n_before - len(ranked)} Ensembl IDs collapsed to a shared base ID after "
            "version stripping; their mean_path_gradient was averaged."
        )

    ranked = ranked.sort_values(RANK_COLUMN, ascending=False, ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ranked.to_csv(out_path, index=False)
    logger.info(f"Ranked gene list ({len(ranked)} genes) -> {out_path}")
    return ranked


def resolve_paths(opt):
    """Resolve (all_genes_csv, out_path) from --in/--out or the config's output_base_dir."""
    if opt.in_csv:
        all_genes_csv = Path(opt.in_csv)
    else:
        from joint_fusion.config.config_manager import ConfigManager

        config = ConfigManager.load_config(opt.config)
        all_genes_csv = (
            Path(config.testing.output_base_dir)
            / "interpret_omics"
            / "all_gene_ig_scores.csv"
        )
    out_path = (
        Path(opt.out) if opt.out else all_genes_csv.parent / "gsea_ranked_genes.csv"
    )
    return all_genes_csv, out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Config used to locate all_gene_ig_scores.csv when --in is absent.",
    )
    parser.add_argument(
        "--in",
        dest="in_csv",
        default=None,
        help="Path to interpret_omics all_gene_ig_scores.csv. Overrides --config.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Default: <interpret_omics dir>/gsea_ranked_genes.csv",
    )
    opt = parser.parse_args()

    all_genes_csv, out_path = resolve_paths(opt)
    if not os.path.exists(all_genes_csv):
        raise SystemExit(
            f"Input not found: {all_genes_csv}\n"
            "Run joint_fusion.testing.interpret_omics first (it writes "
            "all_gene_ig_scores.csv)."
        )
    export_ranked_genes(str(all_genes_csv), str(out_path))


if __name__ == "__main__":
    main()
