import pandas as pd
import os
import pandas as pd
import gget


def clean_ensembl_id(ensembl_id):

    return ensembl_id.split(".")[0]


def get_individual_gene_file_path(output_dir, ensembl_id):
    """
    Get the file path for an individual gene's info file.

    Args:
        output_dir: Directory to save files
        ensembl_id: Ensembl gene ID (will be cleaned of version)

    Returns:
        Path to the gene's individual CSV file
    """
    clean_id = clean_ensembl_id(ensembl_id)
    filename = f"{clean_id}.csv"
    return os.path.join(output_dir, "individual_genes", filename)


def concat_gene_info(
    ensembl_ids,
    output_dir="./gene_info",
    individual_gene_dir="./gene_info/individual_genes",
):

    all_gene_info = []
    missing_files = []

    for ensembl_id in ensembl_ids:
        file_path = get_individual_gene_file_path(individual_gene_dir, ensembl_id)

        if os.path.exists(file_path):
            gene_df = pd.read_csv(file_path)
            all_gene_info.append(gene_df)

        else:
            missing_files.append(ensembl_id)

    if missing_files:
        print(f"Warning: Missing files for {len(missing_files)} genes:")
        for missing_id in missing_files[:10]:
            print(f"  {missing_id}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

    if all_gene_info:
        combined_df = pd.concat(all_gene_info, ignore_index=True)

        # Save combined file
        combined_path = os.path.join(output_dir, "gene_info_combined.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined gene information saved to {combined_path}")

        return combined_df
    else:
        print("No gene info files found to concatenate!")
        return pd.DataFrame()


def obtain_gene_info(ensembl_ids, output_dir="./gene_info/individual_genes"):
    """
    Obtain gene information from Ensembl IDs and save to CSV.

    Args:
        ensembl_ids: List of Ensembl gene IDs
        output_dir: Directory to save the gene info CSV

    Returns:
        pandas.DataFrame: Gene information with ensembl_id, gene_name, description, etc.

    Note:
        Will iterate one at a time because gget.info is slow
    """

    cleaned_ensembl_ids = []

    for ensembl_id in ensembl_ids:

        cleaned_ensembl_ids.append(clean_ensembl_id(ensembl_id))

    num_ensembl_ids = len(cleaned_ensembl_ids)

    # Create output directory to store each genes .csv
    os.makedirs(output_dir, exist_ok=True)

    for idx, ensembl_id in enumerate(cleaned_ensembl_ids):

        print(f"Obtaining gene information for {ensembl_id}: {idx}/{num_ensembl_ids}")

        gene_info_path = os.path.join(output_dir, f"{ensembl_id}.csv")
        if os.path.exists(gene_info_path):
            continue
        else:
            gene_info = gget.info(ensembl_id)

            if gene_info is not None and not gene_info.empty:
                gene_info.to_csv(gene_info_path, index=False)
            else:
                print(f"Warning: No gene info returned for gene {ensembl_id}")

    gene_info_df = concat_gene_info(cleaned_ensembl_ids)

    return gene_info_df


def load_gene_names_from_mapping(
    mapping_file_path="/lus/eagle/clone/g2/projects/GeomicVar/jozhw/multimodal_learning_T1/joint_fusion/mapping_df.json",
    output_dir="./gene_info",
):
    """
    Load gene names from the original mapping data and verify consistency across samples.

    Returns:
        tuple: (gene_names, gene_info_df, is_consistent)
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

    # Obtain gene information
    print("\nObtaining gene information...")
    gene_info_df = obtain_gene_info(gene_names, output_dir)

    return gene_names, gene_info_df, is_consistent


if __name__ == "__main__":

    mapping_file_path = "/lus/eagle/clone/g2/projects/GeomicVar/jozhw/multimodal_learning_T1/joint_fusion/mapping_df.json"
    output_dir = "./gene_info"

    load_gene_names_from_mapping(
        mapping_file_path=mapping_file_path, output_dir=output_dir
    )
