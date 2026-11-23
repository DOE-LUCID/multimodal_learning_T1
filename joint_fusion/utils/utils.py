import os
import re
import gc
from pathlib import Path

import torch
import random
from torch.utils.data._utils.collate import *
from torch.utils.data.dataloader import default_collate


def mixed_collate(batch):
    first_sample = batch[0]

    if len(first_sample) == 5:
        patient_ids, days, events, images_lists, x_omics = zip(*batch)

        days = torch.tensor(days, dtype=torch.float32)
        events = torch.tensor(events, dtype=torch.float32)
        x_omics = torch.stack(x_omics)

        x_wsi = torch.stack(
            [torch.stack(tiles, dim=0) for tiles in images_lists], dim=0
        )

        return patient_ids, days, events, x_wsi, x_omics

    elif len(first_sample) == 6:
        patient_ids, days, events, images_lists, x_omics, tile_names = zip(*batch)

        days = torch.tensor(days, dtype=torch.float32)
        events = torch.tensor(events, dtype=torch.float32)
        x_omics = torch.stack(x_omics)

        x_wsi = torch.stack(
            [torch.stack(tiles, dim=0) for tiles in images_lists], dim=0
        )

        return patient_ids, days, events, x_wsi, x_omics, tile_names

    else:
        raise ValueError(
            f"Unexpected sample size {len(first_sample)} in batch. "
            "Expected 5 or 7 elements per sample."
        )


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def monitor_memory(location=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # gb
        reserved = torch.cuda.memory_reserved() / (1024**3)  # gb
        print(
            f"[{location}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)  # obtain init from global that was set
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)  # If worker does any torch ops


def print_gpu_memory_usage():
    """Print detailed GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")


# Obtaining tile coordinates from tile name
def parse_tile_coordinates(tile_name):
    base = os.path.splitext(os.path.basename(tile_name))[0]
    parts = base.split("-")

    if len(parts) < 2:
        raise ValueError(f"Unexpected tile format: {tile_name}")

    try:
        x = int(parts[-2])
        y = int(parts[-1])
    except ValueError:
        raise ValueError(f"Could not parse coordinates from title name: {tile_name}")

    return x, y
