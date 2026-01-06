# Code to i) create files containing data mapping information (for all modes of training), and ii) joint fusion
# Note: for training using early fusion, refer to early_fusion_survival.py

import pandas as pd
import torch
import h5py
from pdb import set_trace
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from sklearn.model_selection import train_test_split, KFold
import argparse
from concurrent.futures import ThreadPoolExecutor
import pickle
import os
import ast
from collections import Counter

# from data_mapping import create_data_mapping
# for profiling
# torch.autograd.profiler.profile(enabled=True)
from torch.profiler import profile, record_function, ProfilerActivity
from PIL import Image
from sklearn.model_selection import train_test_split

from train_test import train_nn, train_and_test

# on Dell laptop (activate conda env 'pytorch_py3p10' and use 'python trainer.py')
# on Polaris, activate env /lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/pytorch_py3p10


def args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only_create_new_data_mapping",
        action="store_true",
        help="whether to run this script only to create data mapping files and not for training",
    )
    parser.add_argument(
        "--create_new_data_mapping",
        action="store_true",
        help="whether to create new data mapping or use existing one",
    )
    parser.add_argument(
        "--create_new_data_mapping_h5",
        action="store_true",
        help="whether to create new HDF5 data mapping or use existing one",
    )
    parser.add_argument(
        "--input_mapping_data_path",
        type=str,
        default="/lus/eagle/clone/g2/projects/GeomicVar/jozhw/multimodal_learning_T1/joint_fusion/",
        help="Path to input mapping data file",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/",
        help="Path to input data files",
    )
    parser.add_argument(
        "--input_wsi_path",
        type=str,
        default="/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_13july/tiles/256px_9.9x/combined/",
        help="Path to input WSI tiles",
    )
    parser.add_argument(
        "--input_wsi_embeddings_path",
        type=str,
        default="/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_otsu_B/tiles/256px_9.9x/combined_tiles/",
        # on Polaris
        help="Path to WSI embeddings generated from pretrained pathology foundation model",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/joint_fusion/checkpoint_2024-04-20-08-43-52/",
        help="Path to the checkpoint files from trained VAE for omic embedding generation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training (overall)"
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=70,
        help="Batch size for validation data (using all samples for better Cox loss calculation)",
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=1, help="Batch size for testing"
    )
    parser.add_argument(
        "--n_folds", type=int, default=5, help="Number of folds for k-fold CV"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument(
        "--num_epochs", type=int, default=500, help="Number of training epochs"
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",
    )
    parser.add_argument(
        "--input_size_wsi", type=int, default=256, help="input_size for path images"
    )
    parser.add_argument(
        "--embedding_dim_wsi",
        type=int,
        default=384,
        help="embedding dimension for WSI",
    )
    parser.add_argument(
        "--embedding_dim_omic",
        type=int,
        default=256,
        help="embedding dimension for omic (the intermediate layer dim is hardcoded)",
    )
    parser.add_argument(
        "--input_mode", type=str, default="wsi_omic", help="wsi, omic, wsi_omic"
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="joint",
        help="early, late, joint, joint_omic, unimodal",
    )  # "joint_omic" only trains the omic embedding generator jointly with the downstream combined model
    parser.add_argument(
        "--profile", action="store_true", help="Enable profiling (default: False)"
    )
    parser.add_argument(
        "--mlp_layers",
        type=int,
        default=4,
        help="Joint mlp layer number of layers godes from embedding_dim -> 256 -> 256 * (1/2) -> 256 * (1/2)^2 -> .... -> 256 * (1/2)^n -> 1. Example 4 layers means embedding -> 256 -> 128 -> 1",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--use_mixed_precision",
        action="store_true",
        help="whether to use mixed precision calculations (currently unstable)",
    )
    parser.add_argument(
        "--use_gradient_accumulation",
        action="store_true",
        help="whether to use gradient accumulation",
    )

    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Current timestamp that will be used for model identification and labeling. Should be in the format YYYY-MM-DD-HH-MM-SS",
    )

    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage during training",
    )

    ################ Pretraining model access

    parser.add_argument(
        "--use_pretrained_omic",
        action="store_true",
        help="Use pretrained VAE for omic embeddings",
    )

    parser.add_argument(
        "--omic_checkpoint_path",
        type=str,
        default="checkpoints/checkpoint_2024-09-04-07-56-47/checkpoint_epoch_1500.pth",
        help="Path to pretrained omic VAE checkpoint",
    )

    ################ NOTE: These additions are for finding the ideal hyperparams and setup

    # NOTE: Joint embedding only works if model is joint
    parser.add_argument(
        "--joint_embedding",
        type=str,
        default="weighted_avg",
        help="Joint embedding creation method for joint fusion. Current options are concatenate, weighted_avg, and weighted_avg_dynamic",
    )

    ################ NOTE: Create subparsers for learning schedulers

    subparsers = parser.add_subparsers(dest="scheduler", help="Learning rate schedule")

    # Exponential scheduler
    exp_parser = subparsers.add_parser(
        "exponential", help="Exponential learning rate schedule"
    )
    exp_parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        dest="exp_gamma",
        help="Multiplicative factor for learning rate decay",
    )

    # Cosine warmer scheduler
    cosw_parser = subparsers.add_parser(
        "cosine_warmer", help="Cosine annealing learning rate with warmer."
    )
    cosw_parser.add_argument(
        "--t_0", type=int, default=10, help="Number of iterations for the first restart"
    )
    cosw_parser.add_argument(
        "--t_mult", type=int, default=2, help="Cycle length multiplier."
    )
    cosw_parser.add_argument(
        "--eta_min", type=float, default=1e-6, help="Minimum learning rate"
    )

    cosw_parser.add_argument(
        "--decay_factor",
        type=float,
        default=0.5,
        help="Factor by which max learning rate is reduced after each cycle (1.0 = no decay, 0.5 = half each cycle)",
    )

    # Step scheduler
    step_parser = subparsers.add_parser("step_lr", help="Step based learning rate")
    step_parser.add_argument("--step", dest="step_lr_step", type=int, default=10)
    step_parser.add_argument("--gamma", dest="step_lr_gamma", type=float, default=0.95)

    return parser.parse_args()


def create_h5_file(file_name, train_df, val_df, test_df, image_dir):
    with h5py.File(file_name, "w") as hdf:
        for df, split in zip([train_df, val_df, test_df], ["train", "val", "test"]):
            split_group = hdf.create_group(split)
            total_rows = len(df)
            count_row = 0
            for idx, row in df.iterrows():
                # set_trace()
                count_row += 1
                print(f"Processing {split} data: {count_row}/{total_rows} rows")
                patient_group = split_group.create_group(idx)
                patient_group.create_dataset("days_to_death", data=row["days_to_death"])
                patient_group.create_dataset(
                    "days_to_last_followup", data=row["days_to_last_followup"]
                )
                patient_group.create_dataset("days_to_event", data=row["time"])
                patient_group.create_dataset(
                    "event_occurred", data=1 if row["event_occurred"] == "Dead" else 0
                )

                rnaseq_data = np.array(list(row["rnaseq_data"].values()))
                patient_group.create_dataset("rnaseq_data", data=rnaseq_data)

                # store image tiles
                images_group = patient_group.create_group("images")
                for i, tile in enumerate(row["tiles"]):
                    image_path = os.path.join(image_dir, tile)
                    image = Image.open(image_path)
                    img_arr = np.array(image)
                    images_group.create_dataset(
                        f"image_{i}", data=img_arr, compression="gzip"
                    )


def main():

    opt = args()

    if opt.gpu_ids and opt.gpu_ids != "-1":
        gpu_list = [int(x) for x in opt.gpu_ids.split(",")]
        device = torch.device(f"cuda:{gpu_list[0]}")
        print(f"Using GPUs: {gpu_list}")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    torch.backends.cudnn.benchmark = True  # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

    if opt.create_new_data_mapping:
        # create data mappings
        # read the file containing gene expression and tile image locations for the TCGA-LUAD samples (mapped_data_16March)
        # mapping_df = pd.read_csv(opt.input_path + "mapped_data_21March.csv")

        # mapping_df = pd.read_json(opt.input_path + "mapped_data_8may.json", orient='index') # file generated by create_image_molecular_mapping.py
        mapping_df = pd.read_json(
            opt.input_path + "mapped_data_23july.json", orient="index"
        )
        print("Total number of samples: ", mapping_df.shape[0])
        ids_with_wsi = mapping_df[mapping_df["tiles"].map(len) > 0].index.tolist()
        rnaseq_df = pd.DataFrame(
            mapping_df["rnaseq_data"].to_list(), index=mapping_df.index
        ).transpose()
        print("Are there nans in rnaseq_df: ", rnaseq_df.isna().any().any())
        # df containing entries where both WSI and rnaseq data are available
        mapping_df = mapping_df.loc[ids_with_wsi]
        print(
            "Total number of samples where both rnaseq and wsi data are available: ",
            mapping_df.shape[0],
        )
        # set_trace()
        # remove rows where the number of tiles is different from the standard (to avoid length mismatch issues during batching)
        # edit it to keep only 200 random slides among all the available ones
        mask = mapping_df["tiles"].apply(len) == 200
        mapping_df = mapping_df[mask]

        # set_trace()
        # # check if there are empty (or unexpected number of) rnaseq entries
        # # check why this is happening
        # rnaseq_dict_sizes = mapping_df['rnaseq_data'].apply(lambda x: len(x))
        # print("rnaseq_dict_sizes.value_counts(): ", rnaseq_dict_sizes.value_counts())
        # mapping_df = mapping_df[mapping_df['rnaseq_data'].apply(lambda x: len(x) > 0)]
        # print(mapping_df)
        # save the df for using later, or for KM plots
        # mapping_df.to_csv('mapping_df.csv', index=False)

        # combine 'days_to_death' and 'days_to_last_followup' into a single column
        # assuming that for rows where 'days_to_death' is NaN, 'days_to_last_followup' contains the censoring time
        mapping_df["time"] = mapping_df["days_to_death"].fillna(
            mapping_df["days_to_last_followup"]
        )
        mapping_df = mapping_df.dropna(subset=["time", "event_occurred"])
        # remove that from the wsi and rnaseq combined embeddings too
        # rnaseq_df = rnaseq_df.drop(columns=['TCGA-49-6742'])
        # remove entries with anomalous time_to_death and 'days_to_last_followup' data
        excluded_ids = [
            "TCGA-05-4395",
            "TCGA-86-8281",
        ]  # contains anomalous time to event and censoring data
        mapping_df = mapping_df[~mapping_df.index.isin(excluded_ids)]
        mapping_df.to_json("mapping_df.json", orient="index")
        rnaseq_df.to_json("rnaseq_df.json", orient="index")
    else:
        mapping_df = pd.read_json(
            opt.input_mapping_data_path + "mapping_df.json", orient="index"
        )
        # remove entries with anomalous time_to_death and 'days_to_last_followup' data
        excluded_ids = [
            "TCGA-05-4395",
            "TCGA-86-8281",
        ]  # contains anomalous time to event and censoring data
        mapping_df = mapping_df[~mapping_df.index.isin(excluded_ids)]

    # set_trace()

    mapping_df_train, temp_df = train_test_split(
        mapping_df, test_size=0.3, random_state=40
    )
    mapping_df_val, mapping_df_test = train_test_split(
        temp_df, test_size=0.5, random_state=40
    )

    if opt.create_new_data_mapping_h5:
        # create h5 version of mapping_df for faster IO
        create_h5_file(
            "joint_fusion/mapping_data.h5",
            mapping_df_train,
            mapping_df_val,
            mapping_df_test,
            opt.input_wsi_path,
        )

    if not opt.only_create_new_data_mapping:
        log_dir = f"logs/{opt.timestamp}"

        # train the model
        if opt.profile:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                with record_function("model_train"):
                    model, optimizer = train_nn(
                        opt, "joint_fusion/mapping_data.h5", device
                    )
            print("Finishing profiling...")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            trace_path = os.path.join(log_dir, "trace.json")
            prof.export_chrome_trace(trace_path)
            print(f"Saved profile at {trace_path}")

        else:
            # model, optimizer = train_nn(opt, "joint_fusion/mapping_data.h5", device)
            model, optimizer = train_and_test(
                opt, "joint_fusion/mapping_data.h5", device
            )


if __name__ == "__main__":

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    main()
