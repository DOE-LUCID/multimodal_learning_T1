import argparse
import logging
import os
from collections import OrderedDict
from datetime import datetime

import h5py
import numpy as np
import torch

from joint_fusion.config.config_manager import ConfigManager
from joint_fusion.models.multimodal_network import MultimodalNetwork
from joint_fusion.training.pretraining import create_data_loaders
from joint_fusion.utils.logging import setup_logging

from .test import test_and_interpret


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


def setup_model_and_device(config):
    if config.gpu.gpu_ids == "-1":
        device = torch.device("cpu")
        gpu_ids = []
        logging.info("Using CPU")
    else:
        gpu_ids = [int(x) for x in config.gpu.gpu_ids.split(",") if x.strip()]
        if not gpu_ids:
            gpu_ids = [0]

        available_gpus = torch.cuda.device_count()
        gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < available_gpus]

        if not gpu_ids:
            device = torch.device("cpu")
            logging.warning("No valid GPUs found, falling back to CPU")
        else:
            device = torch.device(f"cuda:{gpu_ids[0]}")
            logging.info(f"Available GPUs: {available_gpus}")
            logging.info(f"Using GPUs: {gpu_ids}")

    return device, gpu_ids


def load_model_with_multi_gpu_support(config, device, gpu_ids):
    model = MultimodalNetwork(
        config,
        embedding_dim_wsi=config.model.embedding_dim_wsi,
        embedding_dim_omic=config.model.embedding_dim_omic,
        mode=config.model.input_mode,
        fusion_type=config.model.fusion_type,
        joint_embedding_type=config.model.joint_embedding,
        mlp_layers=config.model.mlp_layers,
        dropout=config.model.dropout,
    )

    logging.info(f"Loading model from: {config.testing.model_path}")
    checkpoint = torch.load(config.testing.model_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    use_multi_gpu = len(gpu_ids) > 1 and config.gpu.use_multi_gpu

    logging.info(f"Saved model has 'module.' prefix: {has_module_prefix}")
    logging.info(f"Will use multi-GPU: {use_multi_gpu}")

    if has_module_prefix and not use_multi_gpu:
        logging.info("Removing 'module.' prefix from state dict")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        state_dict = new_state_dict

    elif not has_module_prefix and use_multi_gpu:
        logging.info("Adding 'module.' prefix to state dict")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[f"module.{k}"] = v
        state_dict = new_state_dict

    model.to(device)

    if use_multi_gpu:
        torch.cuda.set_device(gpu_ids[0])

    model.load_state_dict(state_dict)
    logging.info("Model loaded successfully")

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"test_{timestamp}.log"

    logging.captureWarnings(True)
    setup_logging(
        log_dir=config.testing.output_base_dir,
        log_level=config.logging.log_level,
        log_name=log_name,
    )

    config.gpu.use_multi_gpu = False

    if config.testing.output_base_dir == "":
        raise ValueError("No output_base_dir provided.")
    os.makedirs(config.testing.output_base_dir, exist_ok=True)

    device, gpu_ids = setup_model_and_device(config)
    model = load_model_with_multi_gpu_support(config, device, gpu_ids)
    model.to(device)
    model.eval()

    _, test_loader = create_data_loaders(config, config.data.h5_file)

    mean_x_omic = compute_mean_omic_from_h5(config.data.h5_file)

    test_and_interpret(
        config=config,
        model=model,
        test_loader=test_loader,
        device=device,
        baseline=mean_x_omic,
    )


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()
