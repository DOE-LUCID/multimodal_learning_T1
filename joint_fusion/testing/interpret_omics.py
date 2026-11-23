"""
interpret_omics.py


Run from the repo root:
  python -m joint_fusion.testing.interpret_omics --config=<config.yaml> --top-n=20
"""

import argparse
import functools
import glob
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from joint_fusion.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_gget():
    try:
        import gget

        return gget
    except Exception as e:  # ImportError, or a broken transitive dependency
        logger.warning(
            f"gget is unavailable ({e}); running in cache-only mode. Cache "
            "misses will be listed in missing_genes.txt for local fetching."
        )
        return None


# One <clean_ensembl_id>.csv per gene, same layout get_gene_info.py writes.
GENE_INFO_CACHE_DIR = "assets/gene_info"


def clean_ensembl_id(ensembl_id):
    """Strip the Ensembl version suffix (ENSG000....17 -> ENSG000...)."""
    return str(ensembl_id).split(".")[0]


# ---------------------------------------------------------------------------
# Load IG attributions + gene names
# ---------------------------------------------------------------------------


def load_gene_names(mapping_file_path):
    """Return the ordered list of gene Ensembl IDs from the mapping JSON.

    Validates that every sample has the same RNA-seq gene order, because IG
    vectors are interpreted positionally.
    """
    logger.info(f"Loading gene names from {mapping_file_path}")
    mapping_df = pd.read_json(mapping_file_path, orient="index")

    first_sample_id = mapping_df.index[0]
    gene_names = list(mapping_df.loc[first_sample_id, "rnaseq_data"].keys())

    for sample_id in mapping_df.index:
        observed = list(mapping_df.loc[sample_id, "rnaseq_data"].keys())
        if observed != gene_names:
            raise ValueError(
                f"Gene order mismatch for sample {sample_id}. "
                "Cannot safely align IG vector positions to gene IDs."
            )

    logger.info(
        f"Found {len(gene_names)} genes with validated order across "
        f"{len(mapping_df)} samples"
    )
    return gene_names


def load_ig_matrix(ig_directory, n_genes, prefix="integrated_grads"):
    """Load every <prefix>_<TCGA>.npy file into a (patients x genes) matrix."""
    files = sorted(glob.glob(os.path.join(ig_directory, f"{prefix}_*.npy")))
    if not files:
        return None, []

    rows, patient_ids = [], []
    for file_path in files:
        tcga_id = os.path.basename(file_path)[len(prefix) + 1 : -len(".npy")]
        values = np.asarray(np.load(file_path)).flatten()

        if len(values) != n_genes:
            logger.warning(
                f"Skipping {tcga_id}: {len(values)} {prefix} values != {n_genes} genes"
            )
            continue

        rows.append(values)
        patient_ids.append(tcga_id)

    if not rows:
        raise ValueError(
            f"No {prefix} file matched the expected gene count ({n_genes})."
        )

    logger.info(f"Loaded {prefix} for {len(patient_ids)} patients x {n_genes} genes")
    return np.vstack(rows), patient_ids


# ---------------------------------------------------------------------------
# Rank genes
# ---------------------------------------------------------------------------


def direction_label(value):
    """Convert a signed model-sensitivity value into a risk-direction label."""
    if value is None or pd.isna(value):
        return "n/a"
    if value > 0:
        return "risk-increasing"
    if value < 0:
        return "risk-decreasing"
    return "n/a"


def contribution_label(value):
    """Convert a signed IG value into patient-level contribution language."""
    if value is None or pd.isna(value):
        return "n/a"
    if value > 0:
        return "risk-up contribution"
    if value < 0:
        return "risk-down contribution"
    return "no contribution"


def rounded(value, digits):
    """Round a scalar while preserving missing values for CSV/JSON output."""
    if value is None or pd.isna(value):
        return np.nan
    return round(float(value), digits)


def optional_bool(value):
    """Return a JSON-serializable bool while preserving missing values."""
    if value is None or pd.isna(value):
        return np.nan
    return bool(value)


def sign_agreement(a, b):
    """Per-gene fraction of patient rows where two gradient signs agree."""
    sign_a = np.sign(a)
    sign_b = np.sign(b)
    valid = (sign_a != 0) & (sign_b != 0)
    denom = valid.sum(axis=0)
    numer = ((sign_a == sign_b) & valid).sum(axis=0)
    return np.divide(
        numer,
        denom,
        out=np.full(a.shape[1], np.nan, dtype=float),
        where=denom > 0,
    )


def compute_gene_stats(
    ig_matrix,
    gene_names,
    path_gradient_matrix=None,
    vanilla_gradient_matrix=None,
):
    """Per-gene attribution statistics across patients."""
    abs_matrix = np.abs(ig_matrix)
    mean_ig = ig_matrix.mean(axis=0)
    mean_abs_ig = abs_matrix.mean(axis=0)

    stats = pd.DataFrame(
        {
            "ensembl_id": gene_names,
            "mean_abs_ig": mean_abs_ig,
            "std_abs_ig": abs_matrix.std(axis=0),
            "median_abs_ig": np.median(abs_matrix, axis=0),
            "mean_ig": mean_ig,
            "std_ig": ig_matrix.std(axis=0),
            "signed_retention": np.abs(mean_ig) / (mean_abs_ig + 1e-12),
            "frac_positive": (ig_matrix > 0).mean(axis=0),
            "frac_negative": (ig_matrix < 0).mean(axis=0),
            "n_patients": ig_matrix.shape[0],
        }
    )

    if path_gradient_matrix is not None:
        stats["mean_path_gradient"] = path_gradient_matrix.mean(axis=0)
        stats["std_path_gradient"] = path_gradient_matrix.std(axis=0)
        stats["frac_path_gradient_positive"] = (path_gradient_matrix > 0).mean(axis=0)
    else:
        stats["mean_path_gradient"] = np.nan
        stats["std_path_gradient"] = np.nan
        stats["frac_path_gradient_positive"] = np.nan

    if vanilla_gradient_matrix is not None:
        stats["mean_vanilla_gradient"] = vanilla_gradient_matrix.mean(axis=0)
        stats["std_vanilla_gradient"] = vanilla_gradient_matrix.std(axis=0)
        stats["frac_vanilla_gradient_positive"] = (vanilla_gradient_matrix > 0).mean(
            axis=0
        )
    else:
        stats["mean_vanilla_gradient"] = np.nan
        stats["std_vanilla_gradient"] = np.nan
        stats["frac_vanilla_gradient_positive"] = np.nan

    if path_gradient_matrix is not None and vanilla_gradient_matrix is not None:
        stats["path_vanilla_sign_agreement"] = sign_agreement(
            path_gradient_matrix, vanilla_gradient_matrix
        )
        path_sign = np.sign(stats["mean_path_gradient"].to_numpy())
        vanilla_sign = np.sign(stats["mean_vanilla_gradient"].to_numpy())
        stats["cohort_path_vanilla_direction_agree"] = (
            (path_sign == vanilla_sign) & (path_sign != 0) & (vanilla_sign != 0)
        )
    else:
        stats["path_vanilla_sign_agreement"] = np.nan
        stats["cohort_path_vanilla_direction_agree"] = np.nan

    # Backward-compatible alias: the primary direction metric is path gradient when
    # available, otherwise vanilla gradient. Signed IG is never used for direction.
    if path_gradient_matrix is not None:
        stats["mean_gradient"] = stats["mean_path_gradient"]
        stats["std_gradient"] = stats["std_path_gradient"]
        stats["frac_gradient_positive"] = stats["frac_path_gradient_positive"]
        stats["direction_source"] = "path_gradient"
    elif vanilla_gradient_matrix is not None:
        stats["mean_gradient"] = stats["mean_vanilla_gradient"]
        stats["std_gradient"] = stats["std_vanilla_gradient"]
        stats["frac_gradient_positive"] = stats["frac_vanilla_gradient_positive"]
        stats["direction_source"] = "vanilla_gradient"
    else:
        stats["mean_gradient"] = np.nan
        stats["std_gradient"] = np.nan
        stats["frac_gradient_positive"] = np.nan
        stats["direction_source"] = "none"

    return stats


def rank_genes(stats, top_n, direction_source):
    """Top genes by magnitude, and (if gradients exist) by direction of effect.

    Returns a list of per-ranking DataFrames, each tagged with a ``ranking`` column:
      magnitude       largest mean |IG| -- the genes the model leans on hardest
      risk_increasing largest positive primary gradient -- raising expression raises risk
      risk_decreasing largest negative primary gradient -- raising expression lowers risk
    """
    frames = []

    magnitude = stats.sort_values("mean_abs_ig", ascending=False, ignore_index=True)
    top = magnitude.head(top_n).copy()
    top.insert(0, "ranking", "magnitude")
    top.insert(0, "rank", range(1, len(top) + 1))
    frames.append(top)

    if direction_source == "none":
        logger.warning(
            "No path_gradients_*.npy or vanilla_gradients_*.npy found: reporting the "
            "magnitude ranking only. "
            "Direction is NOT recoverable from the sign of a cohort-mean-baseline IG."
        )
        return frames

    if direction_source == "vanilla_gradient":
        logger.warning(
            "No path_gradients_*.npy found; using vanilla gradients for direction. "
            "That is a valid local Steyaert-style direction, but the preferred "
            "IG-consistent direction is path_gradients_*."
        )

    by_gradient = stats.sort_values("mean_gradient", ascending=False, ignore_index=True)

    up = by_gradient.head(top_n).copy()
    up.insert(0, "ranking", "risk_increasing")
    up.insert(0, "rank", range(1, len(up) + 1))
    frames.append(up)

    down = by_gradient.tail(top_n).iloc[::-1].reset_index(drop=True)
    down.insert(0, "ranking", "risk_decreasing")
    down.insert(0, "rank", range(1, len(down) + 1))
    frames.append(down)

    return frames


# ---------------------------------------------------------------------------
# Annotate genes from Ensembl IDs (cache first, gget.info for misses)
# ---------------------------------------------------------------------------

# Column preference order within a cached gene CSV.
_SYMBOL_COLS = ("primary_gene_name", "ensembl_gene_name", "gene_name", "symbol")
_DESC_COLS = (
    "ensembl_description",
    "ncbi_description",
    "uniprot_description",
    "description",
)


def _extract_field(row, candidate_cols):
    for col in candidate_cols:
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col].strip()
    return None


def load_or_fetch_gene_csv(ensembl_id, cache_dir=GENE_INFO_CACHE_DIR):
    """Return the full gene-info row (as a 1-row DataFrame) for one gene, or None.

    Reads ``cache_dir/<clean_id>.csv``; on a miss, fetches via ``gget.info`` and
    writes it back in get_gene_info.py's format so both share one cache.
    """
    clean = clean_ensembl_id(ensembl_id)
    fp = os.path.join(cache_dir, f"{clean}.csv")

    if os.path.exists(fp):
        try:
            df = pd.read_csv(fp)
            if not df.empty:
                return df
        except Exception:
            pass  # unreadable -> refetch below

    # Cache miss: fetch a single gene and cache it, mirroring get_gene_info.py.
    # Without gget (offline node) we cannot fetch -> report as missing.
    gget = _get_gget()
    if gget is None:
        return None
    try:
        info = gget.info(clean)
    except Exception as e:
        logger.exception(f"gget.info failed for {clean}: {e}")
        return None

    if info is None or info.empty:
        logger.warning(f"No gene info returned for {clean}")
        return None

    info.to_csv(fp, index=False)
    logger.info(f"Fetched and cached gene info -> {fp}")
    return info


def _row_to_details(row):
    """Pull the display fields (symbol/description/biotype) from one gene row."""
    symbol = _extract_field(row, _SYMBOL_COLS)
    if symbol and symbol.startswith("ENSG"):
        symbol = None  # not a real HGNC symbol
    return {
        "symbol": symbol,
        "description": _extract_field(row, _DESC_COLS),
        "biotype": _extract_field(row, ("biotype",)),
    }


def annotate_genes(ensembl_ids, cache_dir=GENE_INFO_CACHE_DIR):
    """Resolve full specifications for each gene from the on-disk cache.

    Returns (details, full_info_df, missing_ids): ``details`` maps ensembl_id to
    {symbol, description, biotype}, ``full_info_df`` concatenates the resolved
    gget.info rows, and ``missing_ids`` lists the unresolved IDs -- fetch those
    locally with get_gene_info.py and copy their CSVs into ``cache_dir``.
    """
    os.makedirs(cache_dir, exist_ok=True)

    details = {}
    full_rows = []
    missing_ids = []
    n_hit = 0
    for eid in ensembl_ids:
        fp = os.path.join(cache_dir, f"{clean_ensembl_id(eid)}.csv")
        cached = os.path.exists(fp)
        df = load_or_fetch_gene_csv(eid, cache_dir)
        if df is None or df.empty:
            details[eid] = {"symbol": None, "description": None, "biotype": None}
            missing_ids.append(clean_ensembl_id(eid))
            continue
        n_hit += 1 if cached else 0
        full_rows.append(df)
        details[eid] = _row_to_details(df.iloc[0])

    logger.info(
        f"gene-info cache ({cache_dir}): {n_hit} hit, "
        f"{len(ensembl_ids) - n_hit} fetched/missing"
    )

    full_info_df = (
        pd.concat(full_rows, ignore_index=True) if full_rows else pd.DataFrame()
    )
    return details, full_info_df, missing_ids


# ---------------------------------------------------------------------------
# Assemble + export
# ---------------------------------------------------------------------------


def build_table(ranking_frames, details):
    """One tidy row per (ranking, gene)."""
    rows = []
    for frame in ranking_frames:
        for _, r in frame.iterrows():
            eid = r["ensembl_id"]
            info = details.get(eid, {})
            gradient = float(r["mean_gradient"])
            path_gradient = r.get("mean_path_gradient", np.nan)
            vanilla_gradient = r.get("mean_vanilla_gradient", np.nan)
            rows.append(
                {
                    "ranking": r["ranking"],
                    "rank": int(r["rank"]),
                    "ensembl_id": eid,
                    "symbol": info.get("symbol"),
                    "description": info.get("description"),
                    "biotype": info.get("biotype"),
                    "mean_abs_ig": round(float(r["mean_abs_ig"]), 6),
                    "std_abs_ig": round(float(r["std_abs_ig"]), 6),
                    "median_abs_ig": round(float(r["median_abs_ig"]), 6),
                    "direction_source": r.get("direction_source", "none"),
                    # Backward-compatible primary direction columns. These are path
                    # gradients when available, else vanilla gradients.
                    "mean_gradient": rounded(gradient, 8),
                    "std_gradient": rounded(r["std_gradient"], 8),
                    "direction": direction_label(gradient),
                    "frac_gradient_positive": rounded(r["frac_gradient_positive"], 4),
                    "mean_path_gradient": rounded(path_gradient, 8),
                    "std_path_gradient": rounded(r.get("std_path_gradient", np.nan), 8),
                    "path_direction": direction_label(path_gradient),
                    "frac_path_gradient_positive": rounded(
                        r.get("frac_path_gradient_positive", np.nan), 4
                    ),
                    "mean_vanilla_gradient": rounded(vanilla_gradient, 8),
                    "std_vanilla_gradient": rounded(
                        r.get("std_vanilla_gradient", np.nan), 8
                    ),
                    "vanilla_direction": direction_label(vanilla_gradient),
                    "frac_vanilla_gradient_positive": rounded(
                        r.get("frac_vanilla_gradient_positive", np.nan), 4
                    ),
                    "path_vanilla_sign_agreement": rounded(
                        r.get("path_vanilla_sign_agreement", np.nan), 4
                    ),
                    "cohort_path_vanilla_direction_agree": optional_bool(
                        r.get("cohort_path_vanilla_direction_agree", np.nan)
                    ),
                    # Transparency only: the sign is patient-relative under a
                    # cohort-mean baseline, so it is not a risk direction.
                    "mean_signed_ig": round(float(r["mean_ig"]), 6),
                    "signed_retention": round(float(r["signed_retention"]), 4),
                    "frac_ig_positive": round(float(r["frac_positive"]), 4),
                    "n_patients": int(r["n_patients"]),
                }
            )
    return pd.DataFrame(rows)


def build_patient_top_table(
    ig_matrix,
    gene_names,
    patient_ids,
    top_n,
    path_gradient_matrix=None,
    vanilla_gradient_matrix=None,
    omic_matrix=None,
    baseline=None,
):
    """Top per-patient signed IG rows.

    The only place signed IG is a valid direction: each row is one patient's
    expression relative to baseline. Cohort direction still comes from gradients.
    """
    if top_n is None or top_n <= 0:
        return None

    baseline_vec = None
    if baseline is not None:
        baseline_vec = np.asarray(baseline).flatten()
        if len(baseline_vec) != len(gene_names):
            logger.warning(
                f"Ignoring ig_baseline.npy: {len(baseline_vec)} values != "
                f"{len(gene_names)} genes"
            )
            baseline_vec = None

    rows = []
    for patient_index, patient_id in enumerate(patient_ids):
        row = ig_matrix[patient_index]
        top_indices = np.argsort(-np.abs(row))[:top_n]
        for rank, gene_index in enumerate(top_indices, start=1):
            signed_ig = float(row[gene_index])
            path_gradient = (
                float(path_gradient_matrix[patient_index, gene_index])
                if path_gradient_matrix is not None
                else np.nan
            )
            vanilla_gradient = (
                float(vanilla_gradient_matrix[patient_index, gene_index])
                if vanilla_gradient_matrix is not None
                else np.nan
            )
            expression = (
                float(omic_matrix[patient_index, gene_index])
                if omic_matrix is not None
                else np.nan
            )
            baseline_value = (
                float(baseline_vec[gene_index]) if baseline_vec is not None else np.nan
            )
            expression_delta = (
                expression - baseline_value
                if not pd.isna(expression) and not pd.isna(baseline_value)
                else np.nan
            )
            rows.append(
                {
                    "patient_id": patient_id,
                    "rank_by_abs_ig": rank,
                    "ensembl_id": gene_names[gene_index],
                    "signed_ig": rounded(signed_ig, 8),
                    "abs_ig": rounded(abs(signed_ig), 8),
                    "patient_ig_contribution": contribution_label(signed_ig),
                    "path_gradient": rounded(path_gradient, 8),
                    "path_direction": direction_label(path_gradient),
                    "vanilla_gradient": rounded(vanilla_gradient, 8),
                    "vanilla_direction": direction_label(vanilla_gradient),
                    "expression": rounded(expression, 8),
                    "baseline": rounded(baseline_value, 8),
                    "expression_minus_baseline": rounded(expression_delta, 8),
                }
            )

    return pd.DataFrame(rows)


def load_baseline(ig_directory, n_genes):
    path = os.path.join(ig_directory, "ig_baseline.npy")
    if not os.path.exists(path):
        logger.warning(
            "No ig_baseline.npy found; patient_top_genes.csv will omit baseline."
        )
        return None
    baseline = np.asarray(np.load(path)).flatten()
    if len(baseline) != n_genes:
        logger.warning(
            f"Skipping ig_baseline.npy: {len(baseline)} values != {n_genes} genes"
        )
        return None
    return baseline


def validate_optional_patient_matrix(name, matrix, ids, patient_ids):
    if matrix is None:
        return None
    if ids != patient_ids:
        raise ValueError(
            f"{name}_*.npy and integrated_grads_*.npy cover different patients; "
            "re-run test.py so they are written in the same pass."
        )
    return matrix


def export_results(table, full_info_df, output_dir, top_n, patient_table=None):
    """Write top_genes.csv/json and the full gene-specification CSV."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "top_genes.csv")
    table.to_csv(csv_path, index=False)
    logger.info(f"Top gene table -> {csv_path}")

    patient_path = None
    if patient_table is not None and not patient_table.empty:
        patient_path = os.path.join(output_dir, "patient_top_genes.csv")
        patient_table.to_csv(patient_path, index=False)
        logger.info(f"Patient-level signed IG table -> {patient_path}")

    if full_info_df is not None and not full_info_df.empty:
        info_path = os.path.join(output_dir, "top_genes_gene_info.csv")
        full_info_df.to_csv(info_path, index=False)
        logger.info(f"Top gene full specifications -> {info_path}")

    json_path = os.path.join(output_dir, "top_genes.json")
    bundle = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "top_n_per_ranking": top_n,
        "ranking_metrics": {
            "magnitude": "mean |IG| -- how much attribution the gene carries",
            "risk_increasing": "largest positive primary mean gradient -- higher "
            "expression raises predicted risk. Primary means path gradient when "
            "available, otherwise vanilla gradient.",
            "risk_decreasing": "largest negative primary mean gradient -- higher "
            "expression lowers predicted risk. Primary means path gradient when "
            "available, otherwise vanilla gradient.",
        },
        "patient_level_file": patient_path,
        "caveat": (
            "The cohort-mean SIGNED IG is reported but is not a direction: the "
            "cohort-mean baseline makes its sign patient-relative, so it cancels "
            "across patients. Direction comes from the path gradient when available "
            "and vanilla gradients are reported as the local Steyaert-style "
            "comparison. See literature/ig_pathway_design_notes.md."
        ),
        "rankings": {
            name: table[table["ranking"] == name].to_dict("records")
            for name in table["ranking"].unique()
        },
    }
    with open(json_path, "w") as fh:
        json.dump(bundle, fh, indent=2)
    logger.info(f"Top gene bundle -> {json_path}")


def log_table(table):
    """Print a compact human-readable summary of the top genes."""
    headlines = {
        "magnitude": "LARGEST ATTRIBUTION (mean |IG|)",
        "risk_increasing": "RISK-INCREASING (largest positive primary mean gradient)",
        "risk_decreasing": "RISK-DECREASING (largest negative primary mean gradient)",
    }
    for ranking, headline in headlines.items():
        subset = table[table["ranking"] == ranking]
        if subset.empty:
            continue
        logger.info("\n" + "=" * 92)
        logger.info(f"TOP {len(subset)} -- {headline}")
        logger.info("=" * 92)
        logger.info(
            f"{'Rank':<5}{'Symbol':<14}{'Mean |IG|':<12}{'Primary grad':<13}Description"
        )
        for _, r in subset.iterrows():
            desc = (r["description"] or "")[:42]
            logger.info(
                f"{r['rank']:<5}{str(r['symbol'] or r['ensembl_id'])[:13]:<14}"
                f"{r['mean_abs_ig']:<12.6f}{r['mean_gradient']:<+13.6f}{desc}"
            )


def write_missing_genes(missing_ids, output_dir):
    if not missing_ids:
        return None

    os.makedirs(output_dir, exist_ok=True)
    missing_path = os.path.join(output_dir, "missing_genes.txt")
    with open(missing_path, "w") as fh:
        fh.write("\n".join(missing_ids) + "\n")

    logger.error(
        f"{len(missing_ids)} top gene(s) could not be annotated from the cache: "
        f"{', '.join(missing_ids)}"
    )
    logger.error(
        f"Wrote missing Ensembl IDs -> {missing_path}. Fetch them locally with "
        f"get_gene_info.py and copy their CSVs into the gene-info cache."
    )
    return missing_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    mapping_file_path,
    ig_directory,
    output_dir,
    top_n,
    patient_top_n,
    gene_info_cache_dir=GENE_INFO_CACHE_DIR,
):
    """Full pipeline: rank -> annotate -> export."""
    gene_names = load_gene_names(mapping_file_path)
    ig_matrix, patient_ids = load_ig_matrix(ig_directory, len(gene_names))
    if ig_matrix is None:
        raise FileNotFoundError(f"No integrated_grads_*.npy files in {ig_directory}")

    path_gradient_matrix, path_gradient_ids = load_ig_matrix(
        ig_directory, len(gene_names), prefix="path_gradients"
    )
    path_gradient_matrix = validate_optional_patient_matrix(
        "path_gradients", path_gradient_matrix, path_gradient_ids, patient_ids
    )

    vanilla_gradient_matrix, vanilla_gradient_ids = load_ig_matrix(
        ig_directory, len(gene_names), prefix="vanilla_gradients"
    )
    vanilla_gradient_matrix = validate_optional_patient_matrix(
        "vanilla_gradients", vanilla_gradient_matrix, vanilla_gradient_ids, patient_ids
    )

    omic_matrix, omic_ids = load_ig_matrix(
        ig_directory, len(gene_names), prefix="omic_input"
    )
    omic_matrix = validate_optional_patient_matrix(
        "omic_input", omic_matrix, omic_ids, patient_ids
    )
    baseline = load_baseline(ig_directory, len(gene_names))

    if path_gradient_matrix is not None:
        direction_source = "path_gradient"
    elif vanilla_gradient_matrix is not None:
        direction_source = "vanilla_gradient"
    else:
        direction_source = "none"

    stats = compute_gene_stats(
        ig_matrix,
        gene_names,
        path_gradient_matrix=path_gradient_matrix,
        vanilla_gradient_matrix=vanilla_gradient_matrix,
    )
    logger.info(
        "signed-IG retention (|mean IG| / mean |IG|) across all genes: "
        f"median {stats['signed_retention'].median():.3f}, "
        f"90th pct {stats['signed_retention'].quantile(0.9):.3f} "
        "-- low values are expected under a cohort-mean baseline and are why "
        "direction is taken from gradients, not the signed IG."
    )
    logger.info(f"Primary direction source: {direction_source}")

    all_genes_path = os.path.join(output_dir, "all_gene_ig_scores.csv")
    stats.sort_values("mean_abs_ig", ascending=False, ignore_index=True).to_csv(
        all_genes_path, index=False
    )
    logger.info(f"All-gene attribution table ({len(stats)} genes) -> {all_genes_path}")

    ranking_frames = rank_genes(stats, top_n, direction_source)

    ensembl_ids = list(
        dict.fromkeys(eid for frame in ranking_frames for eid in frame["ensembl_id"])
    )
    details, full_info_df, missing_ids = annotate_genes(
        ensembl_ids, gene_info_cache_dir
    )
    write_missing_genes(missing_ids, output_dir)

    table = build_table(ranking_frames, details)
    patient_table = build_patient_top_table(
        ig_matrix,
        gene_names,
        patient_ids,
        top_n=patient_top_n,
        path_gradient_matrix=path_gradient_matrix,
        vanilla_gradient_matrix=vanilla_gradient_matrix,
        omic_matrix=omic_matrix,
        baseline=baseline,
    )
    log_table(table)
    export_results(table, full_info_df, output_dir, top_n, patient_table=patient_table)
    return table


def main(config, opt):
    base_dir = Path(config.testing.output_base_dir)
    run(
        mapping_file_path=opt.mapping_file or config.data.json_file,
        ig_directory=opt.ig_dir or str(base_dir / "IG_6sep"),
        output_dir=str(base_dir / "interpret_omics"),
        top_n=opt.top_n,
        patient_top_n=opt.patient_top_n,
        gene_info_cache_dir=opt.gene_info_dir,
    )
    logger.info(
        "\nDone. See all_gene_ig_scores.csv (every gene), top_genes.csv and "
        "patient_top_genes.csv. For the pathway-level analysis, run "
        "joint_fusion.testing.export_gsea_ranked_genes, then knit "
        "joint_fusion/analysis/fgsea_analysis.rmd."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank genes by their IG attribution and direction of effect."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/lus/eagle/clone/g2/projects/GeomicVar/jozhw/multimodal_learning_T1/checkpoints/checkpoint_2026-02-14-05-44-19/config.yaml",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of genes to report per ranking.",
    )
    parser.add_argument(
        "--patient-top-n",
        type=int,
        default=20,
        help="Number of top |IG| genes to export per patient. Use 0 to disable.",
    )
    parser.add_argument(
        "--ig-dir",
        type=str,
        default=None,
        help="IG directory. Defaults to <output_base_dir>/IG_6sep.",
    )
    parser.add_argument(
        "--mapping-file",
        type=str,
        default=None,
        help="Mapping JSON. Defaults to config.data.json_file.",
    )
    parser.add_argument(
        "--gene-info-dir",
        type=str,
        default=GENE_INFO_CACHE_DIR,
        help="Per-gene <ensembl_id>.csv cache; misses are fetched via gget.info "
        "and written back here.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Lazy import so using this module as a library does not pull in pydantic.
    from joint_fusion.config.config_manager import ConfigManager

    opt = parse_args()
    config = ConfigManager.load_config(opt.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(
        log_dir=config.testing.output_base_dir,
        log_level=config.logging.log_level,
        log_name=f"interpret_omics_{timestamp}.log",
    )

    main(config, opt)
