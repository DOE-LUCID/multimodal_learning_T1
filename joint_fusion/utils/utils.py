from pathlib import Path

import torch
import random
from torch.utils.data._utils.collate import *
from torch.utils.data.dataloader import default_collate


def mixed_collate(batch):
    patient_ids, days, events, images_lists, x_omics = zip(*batch)

    days = torch.tensor(days, dtype=torch.float32)
    events = torch.tensor(events, dtype=torch.float32)
    x_omics = torch.stack(x_omics)

    # images_lists: list of patients -> each has 400 tensors
    num_tiles = len(images_lists[0])  # should be 400

    # Transpose: tiles across patients
    x_wsi = []
    for t in range(num_tiles):
        tile_batch = torch.stack([patient_tiles[t] for patient_tiles in images_lists])
        x_wsi.append(tile_batch)

    return patient_ids, days, events, x_wsi, x_omics


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
