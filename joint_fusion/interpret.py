import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from collections import defaultdict
import pandas as pd
import gget


def obtain_gene_info(ensembl_ids, output_dir, chunk_size=1000):
    """
    Obtain gene information from Ensembl IDs and save to CSV.

    Args:
        ensembl_ids: List of Ensembl gene IDs
        output_dir: Directory to save the gene info CSV
        chunk_size: Number of genes to process at once

    Returns:
        pandas.DataFrame: Gene information with ensembl_id, gene_name, description, etc.
    """
    import pandas as pd
    import gget
    import os
    from pathlib import Path

    species = "homo_sapiens"
    ensembl_len = len(ensembl_ids)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    gene_info_path = os.path.join(output_dir, "gene_info.csv")

    # Check if gene info file already exists
    if os.path.exists(gene_info_path):
        print(f"Loading existing gene information from {gene_info_path}")
        gene_info_df = pd.read_csv(gene_info_path)
        return gene_info_df

    print(f"Obtaining gene information for {ensembl_len} genes...")

    # Create chunks
    chunks = []
    if ensembl_len > chunk_size:
        for i in range(0, ensembl_len, chunk_size):
            if (i + chunk_size) > ensembl_len:
                chunks.append(ensembl_ids[i:])
            else:
                chunks.append(ensembl_ids[i : i + chunk_size])
    else:
        chunks.append(ensembl_ids)

    all_gene_info = []

    for chunk_idx, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} genes)")

        try:
            # Get gene information for this chunk; removed species as not an arg
            gene_info = gget.info(chunk)

            if gene_info is not None and not gene_info.empty:
                all_gene_info.append(gene_info)
            else:
                print(f"Warning: No gene info returned for chunk {chunk_idx + 1}")

        except Exception as e:
            print(f"Error processing chunk {chunk_idx + 1}: {e}")
            # Create a fallback DataFrame for failed chunks
            fallback_df = pd.DataFrame(
                {
                    "ensembl_id": chunk,
                    "gene_name": [f"UNKNOWN_{eid}" for eid in chunk],
                    "description": ["Unknown gene"] * len(chunk),
                }
            )
            all_gene_info.append(fallback_df)

    # Combine all results
    if all_gene_info:
        combined_gene_info = pd.concat(all_gene_info, ignore_index=True)

        # Ensure we have the required columns
        required_columns = ["ensembl_id", "gene_name", "description"]
        for col in required_columns:
            if col not in combined_gene_info.columns:
                if col == "gene_name":
                    combined_gene_info[col] = combined_gene_info.get(
                        "ensembl_id", "UNKNOWN"
                    )
                else:
                    combined_gene_info[col] = "Unknown"

        # Save to CSV
        combined_gene_info.to_csv(gene_info_path, index=False)
        print(f"Gene information saved to {gene_info_path}")

        return combined_gene_info


def load_gene_names_from_mapping(
    mapping_file_path="/lus/eagle/clone/g2/projects/GeomicVar/jozhw/multimodal_learning_T1/joint_fusion/mapping_df.json",
    output_dir="./",
):
    """
    Load gene names from the original mapping data and verify consistency across samples.

    Returns:
        tuple: (gene_names_list, is_consistent)
    """

    print(f"Loading gene names from {mapping_file_path}")
    mapping_df = pd.read_json(mapping_file_path, orient="index")

    # Get gene names from the first sample
    first_sample_id = mapping_df.index[0]
    first_rnaseq_data = mapping_df.loc[first_sample_id, "rnaseq_data"]
    gene_names = list(first_rnaseq_data.keys())

    print(f"Found {len(gene_names)} genes in first sample ({first_sample_id})")
    print(f"First 5 genes: {gene_names[:5]}")
    print(f"Last 5 genes: {gene_names[-5:]}")

    # Verify consistency across all samples
    print("\nVerifying gene name consistency across all samples...")
    inconsistent_samples = []

    for idx, (sample_id, row) in enumerate(mapping_df.iterrows()):
        if idx % 50 == 0:  # Print progress every 50 samples
            print(f"  Checking sample {idx + 1}/{len(mapping_df)}: {sample_id}")

        sample_gene_names = list(row["rnaseq_data"].keys())

        if sample_gene_names != gene_names:
            inconsistent_samples.append(sample_id)
            print(f"WARNING: Inconsistent gene names in sample {sample_id}")

            # Show differences for first few inconsistent samples
            if len(inconsistent_samples) <= 3:
                missing_genes = set(gene_names) - set(sample_gene_names)
                extra_genes = set(sample_gene_names) - set(gene_names)
                if missing_genes:
                    print(f"  Missing genes: {list(missing_genes)[:5]}...")
                if extra_genes:
                    print(f"  Extra genes: {list(extra_genes)[:5]}...")

    if inconsistent_samples:
        print(
            f"\nWARNING: Found {len(inconsistent_samples)} samples with inconsistent gene names:"
        )
        for sample_id in inconsistent_samples[:10]:  # Show first 10
            print(f"  {sample_id}")
        if len(inconsistent_samples) > 10:
            print(f"  ... and {len(inconsistent_samples) - 10} more")
        is_consistent = False
    else:
        print("\n✓ All samples have consistent gene names!")
        is_consistent = True

    # Obtain gene information
    print("\nObtaining gene information...")
    gene_info_df = obtain_gene_info(gene_names, output_dir)

    return gene_names, gene_info_df, is_consistent


def verify_hdf5_gene_order(
    h5_file_path="/lus/eagle/clone/g2/projects/GeomicVar/jozhw/multimodal_learning_T1/joint_fusion/mapping_data.h5",
    mapping_file_path="/lus/eagle/clone/g2/projects/GeomicVar/jozhw/multimodal_learning_T1/joint_fusion/mapping_df.json",
    sample_ids_to_check=5,
):
    """
    Verify that the HDF5 file stores RNA-seq data in the same order as the original mapping data.

    Args:
        h5_file_path: Path to the HDF5 file
        mapping_file_path: Path to the original mapping JSON file
        sample_ids_to_check: Number of random samples to verify

    Returns:
        bool: True if order is consistent, False otherwise
    """
    import pandas as pd
    import h5py
    import random

    print(f"Verifying gene order consistency between HDF5 and original mapping data...")

    # Load original mapping data
    mapping_df = pd.read_json(mapping_file_path, orient="index")

    # Get gene names from original data (they should be in consistent order)
    first_sample_id = mapping_df.index[0]
    gene_names = list(mapping_df.loc[first_sample_id, "rnaseq_data"].keys())

    # Get sample IDs that exist in both datasets
    with h5py.File(h5_file_path, "r") as hdf:
        h5_sample_ids = []
        for split in ["train", "val", "test"]:
            if split in hdf:
                h5_sample_ids.extend(list(hdf[split].keys()))

    # Find common sample IDs
    common_sample_ids = list(set(mapping_df.index) & set(h5_sample_ids))

    if len(common_sample_ids) == 0:
        print("ERROR: No common sample IDs found between mapping data and HDF5 file")
        return False

    print(f"Found {len(common_sample_ids)} common samples")

    # Check a random subset
    samples_to_check = min(sample_ids_to_check, len(common_sample_ids))
    random_samples = random.sample(common_sample_ids, samples_to_check)

    order_consistent = True

    with h5py.File(h5_file_path, "r") as hdf:
        for sample_id in random_samples:
            print(f"  Checking sample: {sample_id}")

            # Get original data (as dict)
            original_rnaseq = mapping_df.loc[sample_id, "rnaseq_data"]

            # Get HDF5 data (as array)
            # Find which split this sample is in
            h5_rnaseq = None
            for split in ["train", "val", "test"]:
                if split in hdf and sample_id in hdf[split]:
                    h5_rnaseq = hdf[split][sample_id]["rnaseq_data"][:]
                    break

            if h5_rnaseq is None:
                print(f"    ERROR: Sample {sample_id} not found in HDF5")
                order_consistent = False
                continue

            # Compare values in order
            original_values = [original_rnaseq[gene] for gene in gene_names]
            h5_values = h5_rnaseq.tolist()

            if len(original_values) != len(h5_values):
                print(
                    f"    ERROR: Length mismatch - Original: {len(original_values)}, HDF5: {len(h5_values)}"
                )
                order_consistent = False
                continue

            # Check if values match (allowing for small floating point differences)
            mismatches = 0
            for i, (orig_val, h5_val) in enumerate(zip(original_values, h5_values)):
                if abs(orig_val - h5_val) > 1e-10:
                    mismatches += 1
                    if mismatches <= 5:  # Show first 5 mismatches
                        print(
                            f"    Mismatch at gene {i} ({gene_names[i]}): {orig_val} vs {h5_val}"
                        )

            if mismatches > 0:
                print(f"    ERROR: {mismatches} value mismatches found")
                order_consistent = False
            else:
                print(f"Order and values consistent")

    if order_consistent:
        print("\nGene order is consistent between HDF5 and original mapping data")
    else:
        print("\nGene order inconsistency detected!")

    return order_consistent


def load_integrated_gradients(
    ig_directory="./IG_6sep",
    mapping_file_path="/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/joint_fusion/mapping_df.json",
    output_dir="./",
):
    """
    Load all integrated gradients files and gene names from the original mapping data.

    Returns:
        tuple: (ig_data dict, gene_names list, gene_info_df, is_consistent bool)
    """
    # Load gene names and gene info from original mapping data
    gene_names, gene_info_df, is_consistent = load_gene_names_from_mapping(
        mapping_file_path, output_dir
    )

    if not is_consistent:
        print("WARNING: Gene names are not consistent across samples!")
        print(
            "Results may not be reliable. Consider investigating the data inconsistency."
        )

    # Load IG files
    ig_files = glob.glob(os.path.join(ig_directory, "integrated_grads_*.npy"))
    ig_data = {}

    print(f"\nLoading IG files from {ig_directory}...")

    for file_path in ig_files:
        # Extract TCGA ID from filename
        filename = os.path.basename(file_path)
        tcga_id = filename.replace("integrated_grads_", "").replace(".npy", "")

        # Load the integrated gradients
        ig_values = np.load(file_path)

        # Debug: Print shape and structure info
        print(f"Patient {tcga_id}: Shape = {ig_values.shape}, Type = {type(ig_values)}")

        # Handle different array structures
        if ig_values.ndim > 1:
            if ig_values.shape[0] == 1:
                ig_values = ig_values.flatten()
            else:
                ig_values = ig_values[0] if ig_values.ndim == 2 else ig_values.flatten()

        ig_values = np.asarray(ig_values).flatten()

        print(f"  After processing: Shape = {ig_values.shape}")
        print(f"  Value range: [{np.min(ig_values):.6f}, {np.max(ig_values):.6f}]")
        print(f"  Non-zero values: {np.count_nonzero(ig_values)}")

        # Verify dimensions match gene names
        if len(ig_values) != len(gene_names):
            print(
                f"WARNING: IG values length ({len(ig_values)}) doesn't match gene names length ({len(gene_names)}) for patient {tcga_id}"
            )
            print("This suggests a mismatch between IG calculation and gene order!")
        else:
            print(f"  ✓ IG dimensions match gene count")

        ig_data[tcga_id] = ig_values

    print(f"\nLoaded IG data for {len(ig_data)} patients")
    print(f"Using {len(gene_names)} gene names")

    return ig_data, gene_names, gene_info_df, is_consistent


def get_display_gene_name(ensembl_id, gene_info_df):
    """
    Get the display name for a gene (primary gene name if available, otherwise Ensembl ID).

    Args:
        ensembl_id: Ensembl gene ID
        gene_info_df: DataFrame with gene information

    Returns:
        str: Display name for the gene
    """
    if gene_info_df is not None:
        gene_row = gene_info_df[gene_info_df["ensembl_id"] == ensembl_id]
        if not gene_row.empty and pd.notna(gene_row.iloc[0]["gene_name"]):
            gene_name = gene_row.iloc[0]["gene_name"]
            # If gene_name is different from ensembl_id, use it
            if gene_name != ensembl_id and gene_name != f"UNKNOWN_{ensembl_id}":
                return gene_name

    # Fallback to Ensembl ID
    return ensembl_id


def create_gene_display_mapping(gene_names, gene_info_df):
    """
    Create a mapping from Ensembl IDs to display names.

    Args:
        gene_names: List of Ensembl gene IDs
        gene_info_df: DataFrame with gene information

    Returns:
        dict: Mapping from ensembl_id to display_name
    """
    gene_display_map = {}
    for ensembl_id in gene_names:
        gene_display_map[ensembl_id] = get_display_gene_name(ensembl_id, gene_info_df)

    return gene_display_map


def create_summary_grid_plot(ig_data, patient_ids, top_genes, output_dir):
    """
    Create a summary grid plot showing all patients in one figure for quick comparison.
    """
    # Set up the subplot grid (adjust based on number of patients)
    n_patients = len(patient_ids)
    if n_patients <= 10:
        rows, cols = 2, 5
    elif n_patients <= 16:
        rows, cols = 4, 4
    else:
        rows, cols = int(np.ceil(np.sqrt(n_patients))), int(
            np.ceil(np.sqrt(n_patients))
        )

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n_patients == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, patient_id in enumerate(patient_ids):
        if i >= len(axes):
            break

        ig_values = ig_data[patient_id]

        # Get indices of top positive and negative genes
        top_positive_idx = np.argsort(ig_values)[-top_genes // 2 :][::-1]
        top_negative_idx = np.argsort(ig_values)[: top_genes // 2]

        # Combine and sort by absolute value for better visualization
        combined_idx = np.concatenate([top_positive_idx, top_negative_idx])
        combined_values = ig_values[combined_idx]

        # Create bar plot
        colors = ["red" if val > 0 else "blue" for val in combined_values]

        axes[i].bar(
            range(len(combined_values)), combined_values, color=colors, alpha=0.7
        )
        axes[i].set_title(f"{patient_id}", fontsize=10, fontweight="bold")
        axes[i].set_ylabel("IG Value", fontsize=8)
        axes[i].axhline(y=0, color="black", linestyle="-", alpha=0.3)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis="both", labelsize=8)

    # Hide unused subplots
    for i in range(n_patients, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.suptitle("Gene Expression (IG) - All Patients Summary", fontsize=14, y=1.02)

    summary_path = os.path.join(output_dir, "all_patients_summary_grid.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Summary grid plot saved to {summary_path}")


def create_patient_bar_graphs(
    ig_data,
    gene_names,
    gene_info_df,
    num_patients=10,
    top_genes=50,
    output_dir="./patient_ig_plots",
):
    """
    Create bar graphs for gene expression (IG values) for specified number of patients.
    Now uses display gene names from gene_info_df.
    """
    # Create gene display mapping
    gene_display_map = create_gene_display_mapping(gene_names, gene_info_df)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Limit to specified number of patients
    patient_ids = list(ig_data.keys())[:num_patients]

    for i, patient_id in enumerate(patient_ids):
        print(f"\nProcessing patient {i+1}/{len(patient_ids)}: {patient_id}")

        # Create a new figure for each patient
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

        ig_values = ig_data[patient_id]
        if len(ig_values) > len(gene_names):
            print(
                f" WARNING: More IG values ({len(ig_values)}) than gene names ({len(gene_names)})"
            )
            ig_values = ig_values[: len(gene_names)]
        elif len(ig_values) < len(gene_names):
            print(
                f" WARNING: Fewer IG values ({len(ig_values)}) than gene names ({len(gene_names)})"
            )
            gene_names_subset = gene_names[: len(ig_values)]
        else:
            gene_names_subset = gene_names

        # Debug information
        print(f"  IG values shape: {ig_values.shape}")
        print(f"  Total genes available: {len(ig_values)}")

        available_genes = len(ig_values)
        if available_genes < top_genes:
            print(f"  Adjusting top_genes from {top_genes} to {available_genes}")
            top_genes_adjusted = available_genes
        else:
            top_genes_adjusted = top_genes

        # Handle case where we have very few genes
        if available_genes < 2:
            print(
                f"  Skipping patient {patient_id}: insufficient gene data ({available_genes} genes)"
            )
            plt.close()
            continue

        # Get indices of top positive and negative genes
        half_genes = max(1, top_genes_adjusted // 2)

        # Get indices of top genes
        top_positive_idx = np.argsort(ig_values)[-half_genes:][::-1]
        remaining_genes = top_genes_adjusted - len(top_positive_idx)
        if remaining_genes > 0:
            top_negative_idx = np.argsort(ig_values)[:remaining_genes]
            combined_idx = np.concatenate([top_positive_idx, top_negative_idx])
        else:
            combined_idx = top_positive_idx

        # Get values and corresponding gene names (display names)
        combined_values = ig_values[combined_idx]
        combined_gene_names = [
            gene_display_map.get(gene_names_subset[idx], gene_names_subset[idx])
            for idx in combined_idx
        ]

        print(f"  Selected {len(combined_values)} genes for visualization")
        print(
            f"  Value range: [{np.min(combined_values):.6f}, {np.max(combined_values):.6f}]"
        )

        # Create bar plot
        colors = ["red" if val > 0 else "blue" for val in combined_values]
        x_positions = range(len(combined_values))

        bars = ax.bar(x_positions, combined_values, color=colors, alpha=0.7)

        ax.set_title(
            f"Gene Expression (IG) - Patient: {patient_id}",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("Genes", fontsize=12)
        ax.set_ylabel("IG Value", fontsize=12)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=1)

        # Set gene names as x-tick labels (using display names)
        if len(combined_gene_names) > 20:
            # For many genes, show every nth gene name
            step = max(1, len(combined_gene_names) // 15)
            ax.set_xticks(x_positions[::step])
            ax.set_xticklabels(
                [
                    combined_gene_names[i]
                    for i in range(0, len(combined_gene_names), step)
                ],
                rotation=45,
                ha="right",
                fontsize=8,
            )
        else:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(combined_gene_names, rotation=45, ha="right", fontsize=9)

        # Add grid and legend
        ax.grid(True, alpha=0.3, axis="y")

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="red", alpha=0.7, label="Positive IG"),
            Patch(facecolor="blue", alpha=0.7, label="Negative IG"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        # Add statistics text box
        pos_count = np.sum(combined_values > 0)
        neg_count = np.sum(combined_values < 0)
        zero_count = np.sum(combined_values == 0)
        max_pos = np.max(combined_values) if pos_count > 0 else 0
        min_neg = np.min(combined_values) if neg_count > 0 else 0

        stats_text = f"Total genes: {available_genes}\n"
        stats_text += f"Positive: {pos_count}, Negative: {neg_count}\n"
        stats_text += f"Zero: {zero_count}\n"
        stats_text += f"Max: {max_pos:.4f}, Min: {min_neg:.4f}"

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        # Save individual patient plot
        save_path = os.path.join(output_dir, f"patient_{patient_id}_ig_bargraph.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved plot to {save_path}")

        plt.close()  # Close the figure to free memory

    print(f"\nCompleted processing {len(patient_ids)} patients")
    print(f"All plots saved to {output_dir}")

    # Also create a summary grid plot for quick overview
    create_summary_grid_plot(ig_data, patient_ids, top_genes, output_dir)


def find_top_genes_across_patients(ig_data, gene_names, gene_info_df, top_n=10):
    """
    Find the top N most positive and most negative genes across all patients.
    Now includes frequency in top 10 and average position metrics.

    Args:
        ig_data: Dictionary with TCGA IDs and IG values
        gene_names: List of gene names (Ensembl IDs)
        gene_info_df: DataFrame with gene information
        top_n: Number of top genes to return
    Returns:
        tuple: (top_positive_genes, top_negative_genes) with enhanced gene info
    """
    # Create gene display mapping
    gene_display_map = create_gene_display_mapping(gene_names, gene_info_df)

    # Aggregate all IG values across patients
    all_patients_data = []
    for patient_id, ig_values in ig_data.items():
        for gene_idx, ig_val in enumerate(ig_values):
            if gene_idx < len(gene_names):  # Safety check
                ensembl_id = gene_names[gene_idx]
                display_name = gene_display_map[ensembl_id]
                all_patients_data.append(
                    {
                        "patient_id": patient_id,
                        "gene_index": gene_idx,
                        "ensembl_id": ensembl_id,
                        "gene_name": display_name,
                        "ig_value": ig_val,
                    }
                )
            else:
                print(
                    f"WARNING: Gene index {gene_idx} exceeds gene_names length for patient {patient_id}"
                )

    df = pd.DataFrame(all_patients_data)

    # Calculate per-patient top 10 rankings
    top_10_stats = calculate_top_10_frequency_and_position(ig_data, gene_names, top_n)

    # Add gene display names to top_10_stats
    top_10_stats["ensembl_id"] = top_10_stats["gene_index"].apply(
        lambda idx: gene_names[idx] if idx < len(gene_names) else f"Unknown_{idx}"
    )
    top_10_stats["display_gene_name"] = top_10_stats["ensembl_id"].apply(
        lambda eid: gene_display_map.get(eid, eid)
    )

    # Calculate mean IG value per gene across all patients
    gene_stats = (
        df.groupby(["gene_index", "ensembl_id", "gene_name"])["ig_value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Merge with top 10 statistics
    gene_stats = gene_stats.merge(
        top_10_stats[
            [
                "gene_index",
                "top_10_count_positive",
                "top_10_count_negative",
                "avg_position_positive",
                "avg_position_negative",
            ]
        ],
        on="gene_index",
        how="left",
    )

    # Fill NaN values for genes that never appeared in top 10
    gene_stats["top_10_count_positive"] = gene_stats["top_10_count_positive"].fillna(0)
    gene_stats["top_10_count_negative"] = gene_stats["top_10_count_negative"].fillna(0)
    gene_stats["avg_position_positive"] = gene_stats["avg_position_positive"].fillna(0)
    gene_stats["avg_position_negative"] = gene_stats["avg_position_negative"].fillna(0)

    # Sort by mean IG value
    gene_stats_sorted = gene_stats.sort_values("mean", ascending=False)

    # Get top positive and negative genes
    top_positive = gene_stats_sorted.head(top_n)
    top_negative = gene_stats_sorted.tail(top_n)

    return top_positive, top_negative


def calculate_top_10_frequency_and_position(ig_data, gene_names, top_n=10):
    """
    Calculate how often each gene appears in top 10 and average position.

    Args:
        ig_data: Dictionary with TCGA IDs and IG values
        gene_names: List of gene names (Ensembl IDs)
        top_n: Number of top genes to consider
    Returns:
        DataFrame with gene_index, ensembl_id, top_10_count, and avg_position
    """
    gene_rankings = {"positive": {}, "negative": {}}

    for patient_id, ig_values in ig_data.items():
        # Convert to series for easier sorting
        ig_series = pd.Series(ig_values)

        # Get top N positive genes (highest values)
        top_positive_indices = ig_series.nlargest(top_n).index.tolist()
        for position, gene_idx in enumerate(top_positive_indices, 1):
            if gene_idx not in gene_rankings["positive"]:
                gene_rankings["positive"][gene_idx] = []
            gene_rankings["positive"][gene_idx].append(position)

        # Get top N negative genes (lowest values)
        top_negative_indices = ig_series.nsmallest(top_n).index.tolist()
        for position, gene_idx in enumerate(top_negative_indices, 1):
            if gene_idx not in gene_rankings["negative"]:
                gene_rankings["negative"][gene_idx] = []
            gene_rankings["negative"][gene_idx].append(position)

    # Calculate statistics
    results = []
    all_gene_indices = set()
    all_gene_indices.update(gene_rankings["positive"].keys())
    all_gene_indices.update(gene_rankings["negative"].keys())

    for gene_idx in all_gene_indices:
        pos_positions = gene_rankings["positive"].get(gene_idx, [])
        neg_positions = gene_rankings["negative"].get(gene_idx, [])

        ensembl_id = (
            gene_names[gene_idx]
            if gene_idx < len(gene_names)
            else f"Unknown_{gene_idx}"
        )

        results.append(
            {
                "gene_index": gene_idx,
                "ensembl_id": ensembl_id,
                "top_10_count_positive": len(pos_positions),
                "top_10_count_negative": len(neg_positions),
                "avg_position_positive": np.mean(pos_positions) if pos_positions else 0,
                "avg_position_negative": np.mean(neg_positions) if neg_positions else 0,
            }
        )

    return pd.DataFrame(results)


def plot_top_genes_summary(
    ig_data,
    gene_names,
    gene_info_df,
    output_path,
    top_n=10,
):
    """
    Create enhanced summary plots for top positive and negative genes across all patients.
    """
    top_positive, top_negative = find_top_genes_across_patients(
        ig_data, gene_names, gene_info_df, top_n
    )

    fig, axes = plt.subplots(2, 2, figsize=(24, 20))

    # Plot top positive genes - Mean IG values
    ax1 = axes[0, 0]
    bars1 = ax1.barh(
        range(len(top_positive)),
        top_positive["mean"],
        color="red",
        alpha=0.7,
        xerr=top_positive["std"],
    )
    ax1.set_yticks(range(len(top_positive)))
    ax1.set_yticklabels(
        [
            (
                f"{row['gene_name'][:20]}..."
                if len(row["gene_name"]) > 20
                else row["gene_name"]
            )
            for _, row in top_positive.iterrows()
        ],
        fontsize=10,
    )
    ax1.set_xlabel("Mean IG Value", fontsize=12)
    ax1.set_title(f"Top {top_n} Most Positive Genes - Mean IG Values", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot top positive genes - Top 10 frequency
    ax2 = axes[0, 1]
    bars2 = ax2.barh(
        range(len(top_positive)),
        top_positive["top_10_count_positive"],
        color="darkred",
        alpha=0.7,
    )
    ax2.set_yticks(range(len(top_positive)))
    ax2.set_yticklabels(
        [
            (
                f"{row['gene_name'][:20]}..."
                if len(row["gene_name"]) > 20
                else row["gene_name"]
            )
            for _, row in top_positive.iterrows()
        ],
        fontsize=10,
    )
    ax2.set_xlabel("Times in Top 10 (Positive)", fontsize=12)
    ax2.set_title(f"Top {top_n} Most Positive Genes - Frequency in Top 10", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Plot top negative genes - Mean IG values
    ax3 = axes[1, 0]
    bars3 = ax3.barh(
        range(len(top_negative)),
        top_negative["mean"],
        color="blue",
        alpha=0.7,
        xerr=top_negative["std"],
    )
    ax3.set_yticks(range(len(top_negative)))
    ax3.set_yticklabels(
        [
            (
                f"{row['gene_name'][:20]}..."
                if len(row["gene_name"]) > 20
                else row["gene_name"]
            )
            for _, row in top_negative.iterrows()
        ],
        fontsize=10,
    )
    ax3.set_xlabel("Mean IG Value", fontsize=12)
    ax3.set_title(f"Top {top_n} Most Negative Genes - Mean IG Values", fontsize=14)
    ax3.grid(True, alpha=0.3)

    # Plot top negative genes - Top 10 frequency
    ax4 = axes[1, 1]
    bars4 = ax4.barh(
        range(len(top_negative)),
        top_negative["top_10_count_negative"],
        color="darkblue",
        alpha=0.7,
    )
    ax4.set_yticks(range(len(top_negative)))
    ax4.set_yticklabels(
        [
            (
                f"{row['gene_name'][:20]}..."
                if len(row["gene_name"]) > 20
                else row["gene_name"]
            )
            for _, row in top_negative.iterrows()
        ],
        fontsize=10,
    )
    ax4.set_xlabel("Times in Top 10 (Negative)", fontsize=12)
    ax4.set_title(f"Top {top_n} Most Negative Genes - Frequency in Top 10", fontsize=14)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    return top_positive, top_negative


def print_gene_rankings(top_positive, top_negative):
    """
    Print enhanced formatted lists of top positive and negative genes.
    """
    print("\n" + "=" * 160)
    print("TOP 10 MOST POSITIVE GENES ACROSS ALL PATIENTS")
    print("=" * 160)
    print(
        f"{'Rank':<6} {'Gene Index':<12} {'Ensembl ID':<18} {'Gene Name':<25} {'Mean IG':<12} {'Std IG':<12} "
        f"{'Count':<8} {'Top10 Freq':<12} {'Avg Pos':<10}"
    )
    print("-" * 160)

    for rank, (_, row) in enumerate(top_positive.iterrows(), 1):
        ensembl_display = (
            row["ensembl_id"][:17] if len(row["ensembl_id"]) > 17 else row["ensembl_id"]
        )
        gene_name_display = (
            row["gene_name"][:24] if len(row["gene_name"]) > 24 else row["gene_name"]
        )
        print(
            f"{rank:<6} {int(row['gene_index']):<12} {ensembl_display:<18} {gene_name_display:<25} "
            f"{row['mean']:<12.6f} {row['std']:<12.6f} {row['count']:<8} "
            f"{int(row['top_10_count_positive']):<12} {row['avg_position_positive']:<10.2f}"
        )

    print("\n" + "=" * 160)
    print("TOP 10 MOST NEGATIVE GENES ACROSS ALL PATIENTS")
    print("=" * 160)
    print(
        f"{'Rank':<6} {'Gene Index':<12} {'Ensembl ID':<18} {'Gene Name':<25} {'Mean IG':<12} {'Std IG':<12} "
        f"{'Count':<8} {'Top10 Freq':<12} {'Avg Pos':<10}"
    )
    print("-" * 160)

    # Reverse order for negative genes (most negative first)
    negative_rows = list(top_negative.iterrows())
    for rank, (_, row) in enumerate(reversed(negative_rows), 1):
        ensembl_display = (
            row["ensembl_id"][:17] if len(row["ensembl_id"]) > 17 else row["ensembl_id"]
        )
        gene_name_display = (
            row["gene_name"][:24] if len(row["gene_name"]) > 24 else row["gene_name"]
        )
        print(
            f"{rank:<6} {int(row['gene_index']):<12} {ensembl_display:<18} {gene_name_display:<25} "
            f"{row['mean']:<12.6f} {row['std']:<12.6f} {row['count']:<8} "
            f"{int(row['top_10_count_negative']):<12} {row['avg_position_negative']:<10.2f}"
        )


def analyze_gene_consistency(ig_data, gene_names, gene_info_df, top_n=10):
    """
    Additional analysis function to show most consistently ranked genes.
    """
    top_positive, top_negative = find_top_genes_across_patients(
        ig_data, gene_names, gene_info_df, top_n
    )

    print("\n" + "=" * 80)
    print("MOST CONSISTENTLY HIGH-RANKING POSITIVE GENES")
    print("=" * 80)

    # Sort by frequency in top 10, then by average position
    consistent_positive = top_positive.sort_values(
        ["top_10_count_positive", "avg_position_positive"], ascending=[False, True]
    ).head(10)

    print(
        f"{'Gene Name':<25} {'Times in Top10':<15} {'Avg Position':<15} {'Mean IG':<12}"
    )
    print("-" * 80)
    for _, row in consistent_positive.iterrows():
        gene_name_display = (
            row["gene_name"][:24] if len(row["gene_name"]) > 24 else row["gene_name"]
        )
        print(
            f"{gene_name_display:<24} {int(row['top_10_count_positive']):<15} "
            f"{row['avg_position_positive']:<15.2f} {row['mean']:<12.6f}"
        )

    print("\n" + "=" * 80)
    print("MOST CONSISTENTLY HIGH-RANKING NEGATIVE GENES")
    print("=" * 80)

    consistent_negative = top_negative.sort_values(
        ["top_10_count_negative", "avg_position_negative"], ascending=[False, True]
    ).head(10)

    print(
        f"{'Gene Name':<25} {'Times in Top10':<15} {'Avg Position':<15} {'Mean IG':<12}"
    )
    print("-" * 80)
    for _, row in consistent_negative.iterrows():
        gene_name_display = (
            row["gene_name"][:24] if len(row["gene_name"]) > 24 else row["gene_name"]
        )
        print(
            f"{gene_name_display:<24} {int(row['top_10_count_negative']):<15} "
            f"{row['avg_position_negative']:<15.2f} {row['mean']:<12.6f}"
        )


def analyze_patient_gene_patterns(
    ig_data, top_positive, top_negative, output_path="gene_pattern_heatmaps.png"
):
    """
    Analyze how the top genes vary across individual patients.
    """
    # Get gene indices for top positive and negative genes
    top_pos_genes = top_positive["gene_index"].values
    top_neg_genes = top_negative["gene_index"].values

    # Create heatmap data
    patient_ids = list(ig_data.keys())

    # Positive genes heatmap
    pos_data = []
    for patient_id in patient_ids:
        ig_values = ig_data[patient_id]
        patient_pos_values = [ig_values[int(gene_idx)] for gene_idx in top_pos_genes]
        pos_data.append(patient_pos_values)

    # Negative genes heatmap
    neg_data = []
    for patient_id in patient_ids:
        ig_values = ig_data[patient_id]
        patient_neg_values = [ig_values[int(gene_idx)] for gene_idx in top_neg_genes]
        neg_data.append(patient_neg_values)

    # Create heatmaps
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Positive genes heatmap
    sns.heatmap(
        pos_data,
        xticklabels=[f"Gene {int(g)}" for g in top_pos_genes],
        yticklabels=[f"Patient {i+1}" for i in range(len(patient_ids))],
        cmap="Reds",
        center=0,
        ax=ax1,
    )
    ax1.set_title("Top Positive Genes Across Patients")
    ax1.set_ylabel("Patients")

    # Negative genes heatmap
    sns.heatmap(
        neg_data,
        xticklabels=[f"Gene {int(g)}" for g in top_neg_genes],
        yticklabels=[f"Patient {i+1}" for i in range(len(patient_ids))],
        cmap="Blues_r",
        center=0,
        ax=ax2,
    )
    ax2.set_title("Top Negative Genes Across Patients")
    ax2.set_ylabel("Patients")
    ax2.set_xlabel("Genes")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


# Main execution function
def main(opt):
    """
    Main function to run the complete analysis.
    """

    base_dir = Path(opt.output_base_dir)
    ig_directory = str(base_dir / "IG_6sep")

    # Load IG data - NOW ALSO GETS GENE INFO
    ig_data, gene_names, gene_info_df, is_consistent = load_integrated_gradients(
        ig_directory=ig_directory,
        output_dir=opt.output_base_dir,  # Pass output_base_dir for CSV storage
    )

    if not ig_data:
        print("No IG data found. Make sure the IG files exist in ./IG_6sep directory")
        return

    if not is_consistent:
        print("\nWARNING: Gene name inconsistency detected!")
        print("Results may not be reliable. Consider investigating the data.")
        proceed = input("Do you want to proceed anyway? (y/n): ").lower().strip()
        if proceed != "y":
            return

    print(f"\nAnalysis will use {len(gene_names)} genes")

    # Verify HDF5 gene order consistency
    print("\nVerifying HDF5 gene order consistency...")
    verify_hdf5_gene_order()

    # Create bar graphs for first 10 patients - NOW USES GENE INFO
    print("Creating bar graphs for gene expression (IG) for 10 patients...")
    patient_bar_graph_path = str(base_dir / "patient_ig_plots")
    create_patient_bar_graphs(
        ig_data,
        gene_names,
        gene_info_df,
        num_patients=10,
        output_dir=patient_bar_graph_path,
    )

    # Find and plot top genes across all patients - NOW USES GENE INFO
    print("Analyzing top genes across all patients...")
    plot_top_genes_path = str(base_dir / "top_genes_summary.png")
    top_positive, top_negative = plot_top_genes_summary(
        ig_data, gene_names, gene_info_df, top_n=10, output_path=plot_top_genes_path
    )

    # Print gene rankings
    print_gene_rankings(top_positive, top_negative)

    # Create heatmaps showing gene patterns across patients
    print("Creating gene pattern heatmaps...")
    analyze_patient_gene_patterns_path = str(base_dir / "gene_pattern_heatmaps.png")
    analyze_patient_gene_patterns(
        ig_data,
        top_positive,
        top_negative,
        output_path=analyze_patient_gene_patterns_path,
    )
    analyze_gene_consistency(ig_data, gene_names, gene_info_df)

    print("\nAnalysis complete! Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=False,
        help="Indicate the base path for both input files and output files for this analysis.",
    )

    opt = parser.parse_args()

    os.makedirs(opt.output_base_dir, exist_ok=True)

    main(opt)
