import os
import numpy as np
import torch
from torch.utils.data import ConcatDataset

from torch import nn
import h5py
import logging
import argparse
import pandas as pd

from .test import test_and_interpret

from joint_fusion.models.multimodal_network import MultimodalNetwork
from joint_fusion.models.loss_functions import JointLoss

from joint_fusion.training.pretraining import (
    create_data_loaders,
)
from joint_fusion.config.config_manager import ConfigManager

from pathlib import Path


def compute_mean_omic_from_h5(file_name):
    with h5py.File(file_name, "r") as hdf:
        train_group = hdf["train"]
        total_rnaseq_data = None
        total_samples = 0

        for patient_id in train_group.keys():
            patient_group = train_group[patient_id]
            rnaseq_data = patient_group["rnaseq_data"][:]
            if total_rnaseq_data is None:
                total_rnaseq_data = np.zeros_like(rnaseq_data)
            total_rnaseq_data += rnaseq_data
            total_samples += 1
        mean_rnaseq_data = total_rnaseq_data / total_samples

    return mean_rnaseq_data


def load_model_state_dict(model, checkpoint_path):
    """
    Load model state dict, handling DataParallel prefix issues
    """
    from collections import OrderedDict

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # Get model's expected keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    # Check if we need to add or remove 'module.' prefix
    if len(model_keys & checkpoint_keys) == 0:  # No matching keys
        if any(k.startswith("module.") for k in checkpoint_keys):
            # Remove 'module.' prefix
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            # Add 'module.' prefix
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[f"module.{k}"] = v
            state_dict = new_state_dict

    model.load_state_dict(state_dict)
    return model


def setup_model_and_device(config):
    """Setup model and device configuration"""

    # Parse GPU IDs
    if config.gpu.gpu_ids == "-1":
        device = torch.device("cpu")
        gpu_ids = []
        print("Using CPU")
    else:
        gpu_ids = [int(x) for x in config.gpu.gpu_ids.split(",") if x.strip()]
        if not gpu_ids:
            gpu_ids = [0]

        # Check if requested GPUs are available
        available_gpus = torch.cuda.device_count()
        gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < available_gpus]

        if not gpu_ids:
            device = torch.device("cpu")
            print("No valid GPUs found, falling back to CPU")
        else:
            device = torch.device(f"cuda:{gpu_ids[0]}")
            print(f"Available GPUs: {available_gpus}")
            print(f"Using GPUs: {gpu_ids}")

    return device, gpu_ids


def load_model_with_multi_gpu_support(config, device, gpu_ids):
    """Load model with proper multi-GPU support"""

    # Create base model
    model = MultimodalNetwork(
        embedding_dim_wsi=config.model.embedding_dim_wsi,
        embedding_dim_omic=config.model.embedding_dim_omic,
        mode=config.model.input_mode,
        fusion_type=config.model.fusion_type,
        joint_embedding_type=config.model.joint_embedding,
        mlp_layers=config.model.mlp_layers,
        dropout=config.model.dropout,
    )

    # Load checkpoint
    print(f"Loading model from: {config.testing.model_path}")
    checkpoint = torch.load(config.testing.model_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # Determine if the saved model was using DataParallel
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    use_multi_gpu = len(gpu_ids) > 1 and config.gpu.use_multi_gpu

    print(f"Saved model has 'module.' prefix: {has_module_prefix}")
    print(f"Will use multi-GPU: {use_multi_gpu}")

    # Handle state dict prefix conversion
    from collections import OrderedDict

    if has_module_prefix and not use_multi_gpu:
        # Remove 'module.' prefix - saved with DataParallel, loading without
        print("Removing 'module.' prefix from state dict")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        state_dict = new_state_dict

    elif not has_module_prefix and use_multi_gpu:
        # Add 'module.' prefix - saved without DataParallel, loading with
        print("Adding 'module.' prefix to state dict")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[f"module.{k}"] = v
        state_dict = new_state_dict

    # Move model to device first
    model.to(device)

    # Apply DataParallel if using multiple GPUs
    if use_multi_gpu:
        # print(f"Wrapping model with DataParallel on GPUs: {gpu_ids}")
        # model = nn.DataParallel(model, device_ids=gpu_ids)
        # Set the primary GPU
        torch.cuda.set_device(gpu_ids[0])

    # Load the state dict
    model.load_state_dict(state_dict)
    print("Model loaded successfully")

    return model


def args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="/lus/eagle/clone/g2/projects/GeomicVar/jozhw/multimodal_learning_T1/joint_fusion/config/base_config.yaml",
    )

    return parser.parse_args()


def main():

    opt = args()
    config = ConfigManager.load_config(opt.config)
    config.gpu.use_multi_gpu = False  # testing does not support multigpu currently
    if config.testing.output_base_dir == "":
        raise ValueError("No output_base_dir inputed.")
    else:
        os.makedirs(config.testing.output_base_dir, exist_ok=True)

    device, gpu_ids = setup_model_and_device(config)
    model = load_model_with_multi_gpu_support(config, device, gpu_ids)
    model.to(device)
    joint_loss = JointLoss()

    train_loader, validation_loader, test_loader = create_data_loaders(
        config, config.data.input_mapping_data_path + "mapping_data.h5"
    )
    validation_dataset = validation_loader.dataset
    test_dataset = test_loader.dataset

    # create a combined loader (validation + test) as the validation data hasn't been used for HPO during training
    combined_dataset = ConcatDataset([validation_dataset, test_dataset])
    combined_loader = torch.utils.data.DataLoader(
        dataset=combined_dataset,
        batch_size=config.testing.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    mean_x_omic = compute_mean_omic_from_h5(
        config.data.input_mapping_data_path + "mapping_data.h5"
    )

    test_and_interpret(config, model, combined_loader, device, baseline=mean_x_omic)


if __name__ == "__main__":

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    main()
