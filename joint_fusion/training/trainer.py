# Code to i) create files containing data mapping information (for all modes of training), and ii) joint fusion
# Note: for training using early fusion, refer to early_fusion_survival.py

import pandas as pd
import torch
import h5py
import numpy as np
import random
import ast
import time

import os
import argparse

from torch.profiler import profile, record_function, ProfilerActivity
from PIL import Image
from sklearn.model_selection import train_test_split

from .train_observ_test import train_observ_test
from .train import train_nn

from joint_fusion.config.config_manager import ConfigManager
from joint_fusion.utils.logging import setup_logging
import logging
import warnings


def args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="/lus/eagle/clone/g2/projects/GeomicVar/jozhw/multimodal_learning_T1/joint_fusion/config/base_config.yaml",
    )

    return parser.parse_args()


def set_global_seed(seed: int, deterministic: bool = True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(gpu_ids: str):
    if gpu_ids and gpu_ids != "-1":
        gpu_list = [int(x) for x in gpu_ids.split(",")]
        logging.info(f"Using GPUs: {gpu_list}")
        return torch.device(f"cuda:{gpu_list[0]}")
    return torch.device("cpu")


def sample_tiles_per_slide(mapping_df, n_tiles: int, seed: int):
    rng = random.Random(seed)

    def _sample(tiles):
        if len(tiles) < n_tiles:
            return None
        return rng.sample(tiles, n_tiles)

    mapping_df = mapping_df.copy()
    mapping_df["tiles"] = mapping_df["tiles"].apply(_sample)
    mapping_df = mapping_df.dropna(subset=["tiles"])

    return mapping_df


def create_mapping_df(
    input_json: str,
    n_tiles: int,
    seed: int,
    excluded_ids: list[str],
):

    mapping_df = pd.read_json(input_json, orient="index")
    logging.info(f"Loaded mapping: {mapping_df.shape[0]} samples")

    if isinstance(mapping_df["rnaseq_data"].iloc[0], str):
        logging.info("Parsing rnaseq_data from string format...")
        mapping_df["rnaseq_data"] = mapping_df["rnaseq_data"].apply(ast.literal_eval)

    ids_with_wsi = mapping_df[mapping_df["tiles"].map(len) > 0].index.tolist()

    rnaseq_df = pd.DataFrame(
        mapping_df["rnaseq_data"].to_list(), index=mapping_df.index
    ).transpose()

    logging.warning("Are there nans in rnaseq_df: %s", rnaseq_df.isna().any().any())

    mapping_df = mapping_df.loc[ids_with_wsi]

    logging.info(
        "Total number of samples where both rnaseq and wsi data are available: %s",
        mapping_df.shape[0],
    )

    # sample tiles
    mapping_df = sample_tiles_per_slide(mapping_df, n_tiles, seed)

    # survival time
    mapping_df["time"] = mapping_df["days_to_death"].fillna(
        mapping_df["days_to_last_followup"]
    )

    mapping_df = mapping_df.dropna(subset=["time", "event_occurred"])
    mapping_df = mapping_df[~mapping_df.index.isin(excluded_ids)]

    logging.info(f"Final mapping size: {mapping_df.shape[0]}")
    return mapping_df


def split_mapping(mapping_df, seed: int):
    train_df, temp_df = train_test_split(mapping_df, test_size=0.3, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)
    return train_df, val_df, test_df


def create_h5_file(file_name, train_df, val_df, test_df, image_dir):
    logger = logging.getLogger("h5_creation")

    start_time = time.time()

    temp_file = file_name + ".tmp"

    logger.info(f"Creating HDF5 file (temp): {temp_file}")

    with h5py.File(temp_file, "w") as hdf:
        for df, split in zip([train_df, val_df, test_df], ["train", "val", "test"]):

            logger.info(f"Creating split group: {split} ({len(df)} patients)")
            split_group = hdf.create_group(split)

            total_rows = len(df)
            count_row = 0
            for idx, row in df.iterrows():
                # set_trace()
                count_row += 1

                logger.info(f"Processing {split} data: {count_row}/{total_rows} rows")

                patient_group = split_group.create_group(idx)
                patient_group.create_dataset("days_to_death", data=row["days_to_death"])
                patient_group.create_dataset(
                    "days_to_last_followup", data=row["days_to_last_followup"]
                )
                patient_group.create_dataset("days_to_event", data=row["time"])
                patient_group.create_dataset(
                    "event_occurred", data=1 if row["event_occurred"] == "Dead" else 0
                )

                rnaseq = row["rnaseq_data"]
                rnaseq_data = np.array(list(rnaseq.values()))

                patient_group.create_dataset("rnaseq_data", data=rnaseq_data)

                # store image tiles
                images_group = patient_group.create_group("images")
                for i, tile in enumerate(row["tiles"]):
                    image_path = os.path.join(image_dir, tile)
                    with Image.open(image_path) as image:
                        img_arr = np.array(image)
                        images_group.create_dataset(
                            f"image_{i}", data=img_arr, compression="gzip"
                        )

            logger.info(f"Finished split: {split}")

        hdf.flush()
        logger.info("Flushed HDF5 file to disk")

    logger.info("Validating temporary HDF5 file")

    with h5py.File(temp_file, "r") as f:
        for split in ["train", "val", "test"]:
            logger.info(f" {split}: {len(f[split])} patients")

    os.replace(temp_file, file_name)
    elapsed = time.time() - start_time
    logger.info(f"HDF5 creation complete: {file_name} ({elapsed:.1f})s")


def main():

    opt = args()

    config = ConfigManager.load_config(opt.config)

    # make model deterministic for reproducibility
    seed = config.training.random_state

    set_global_seed(seed, deterministic=True)

    os.makedirs(config.logging.checkpoint_dir, exist_ok=True)
    ConfigManager.save_config(config, config.logging.checkpoint_dir)

    logging.captureWarnings(True)

    logger = setup_logging(
        log_dir=config.logging.checkpoint_dir,
        log_level=config.logging.log_level,
    )

    h5_logger = logging.getLogger("h5_creation")
    h5_logger.setLevel(logging.INFO)

    logger.info("Starting training")
    logger.info(f"Checkpoint dir: {config.logging.checkpoint_dir}")

    device = get_device(config.gpu.gpu_ids)

    logging.info(f"Using device: {device}")
    # torch.backends.cudnn.benchmark = True  # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

    excluded_ids = [
        "TCGA-05-4395",
        "TCGA-86-8281",
    ]  # contains anomalous time to event and censoring data
    if config.data.create_new_data_mapping:

        mapping_df = create_mapping_df(
            input_json=config.data.input_base_mapping_data_json_path,
            n_tiles=config.data.num_tiles_per_slide,
            seed=config.training.random_state,
            excluded_ids=excluded_ids,
        )

        mapping_df.to_json(
            os.path.join(config.data.input_base_path, "mapping_df.json"),
            orient="index",
        )

    else:
        mapping_df = pd.read_json(
            os.path.join(config.data.input_base_path, "mapping_df.json"),
            orient="index",
        )

    # set_trace()

    train_df, val_df, test_df = split_mapping(mapping_df, config.training.random_state)

    if config.data.create_new_data_mapping_h5:
        # create h5 version of mapping_df for faster IO
        create_h5_file(
            os.path.join(config.data.input_base_path, "mapping_data.h5"),
            train_df,
            val_df,
            test_df,
            config.data.input_wsi_path,
        )

    if not config.data.only_create_new_data_mapping:

        # train the model
        if config.logging.profile:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                with record_function("model_train"):
                    model, optimizer = train_nn(
                        config, "joint_fusion/mapping_data.h5", device
                    )
            logging.info("Finishing profiling...")
            logging.info(
                prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
            )
            trace_path = os.path.join(config.logging.checkpoint_dir, "trace.json")
            prof.export_chrome_trace(trace_path)
            logging.info(f"Saved profile at {trace_path}")

        else:
            # model, optimizer = train_nn(config, config.data.h5_file, device)
            # used to observe test set to see if training is stable
            model, optimizer = train_observ_test(config, config.data.h5_file, device)


if __name__ == "__main__":

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    main()
