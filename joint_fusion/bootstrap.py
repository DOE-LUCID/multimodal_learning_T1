import argparse
import json
import logging
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sksurv.metrics import concordance_index_censored

from joint_fusion.config.config_manager import ConfigManager
from joint_fusion.models.multimodal_network import MultimodalNetwork
from joint_fusion.training.pretraining import create_data_loaders
from joint_fusion.utils.logging import setup_logging


logger = logging.getLogger(__name__)


def args():
    parser = argparse.ArgumentParser(
        description="Bootstrap the current joint-fusion test predictions."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="joint_fusion/config/base_config.yaml",
        help="Path to the current joint-fusion config YAML.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=100,
        help="Number of bootstrap iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=422,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Bootstrap sample size. Defaults to the number of evaluated test samples.",
    )
    return parser.parse_args()


def setup_model_and_device(config):
    if config.gpu.gpu_ids == "-1":
        device = torch.device("cpu")
        gpu_ids = []
        logger.info("Using CPU")
    else:
        gpu_ids = [int(x) for x in config.gpu.gpu_ids.split(",") if x.strip()]
        if not gpu_ids:
            gpu_ids = [0]

        available_gpus = torch.cuda.device_count()
        gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < available_gpus]

        if not gpu_ids:
            device = torch.device("cpu")
            logger.warning("No valid GPUs found, falling back to CPU")
        else:
            device = torch.device(f"cuda:{gpu_ids[0]}")
            logger.info("Available GPUs: %s", available_gpus)
            logger.info("Using GPUs: %s", gpu_ids)

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

    logger.info("Loading model from: %s", config.testing.model_path)
    checkpoint = torch.load(config.testing.model_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    use_multi_gpu = len(gpu_ids) > 1 and config.gpu.use_multi_gpu

    logger.info("Saved model has 'module.' prefix: %s", has_module_prefix)
    logger.info("Will use multi-GPU: %s", use_multi_gpu)

    if has_module_prefix and not use_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        state_dict = new_state_dict
    elif not has_module_prefix and use_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[f"module.{k}"] = v
        state_dict = new_state_dict

    model.to(device)

    if use_multi_gpu:
        torch.cuda.set_device(gpu_ids[0])

    model.load_state_dict(state_dict)
    logger.info("Model loaded successfully")
    return model


def unpack_test_batch(batch):
    if len(batch) == 6:
        tcga_id, days_to_event, event_occurred, x_wsi, x_omic, _tile_names = batch
    else:
        tcga_id, days_to_event, event_occurred, x_wsi, x_omic = batch
    return tcga_id, days_to_event, event_occurred, x_wsi, x_omic


def collect_test_predictions(model, test_loader, device, excluded_ids=None):
    if excluded_ids is None:
        excluded_ids = ["TCGA-05-4395", "TCGA-86-8281"]

    all_predictions = []
    all_times = []
    all_events = []
    all_tcga_ids = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            tcga_id, days_to_event, event_occurred, x_wsi, x_omic = unpack_test_batch(
                batch
            )

            if tcga_id[0] in excluded_ids:
                logger.info("Skipping TCGA ID: %s", tcga_id[0])
                continue

            x_wsi = x_wsi.to(device)
            x_omic = x_omic.to(device)

            outputs, _, _, _ = model(
                x_wsi=x_wsi,
                x_omic=x_omic,
                return_attention=False,
            )

            all_predictions.append(float(outputs.squeeze().detach().cpu().item()))
            all_times.append(float(days_to_event.squeeze().detach().cpu().item()))
            all_events.append(int(event_occurred.squeeze().detach().cpu().item()))
            all_tcga_ids.append(tcga_id[0])

            if device.type == "cuda" and (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

    predictions = np.asarray(all_predictions, dtype=np.float32)
    times = np.asarray(all_times, dtype=np.float32)
    events = np.asarray(all_events, dtype=np.int32)

    if len(predictions) == 0:
        raise ValueError("No test predictions were collected.")

    return all_tcga_ids, predictions, times, events


def bootstrap_c_indices(predictions, times, events, n_bootstraps, sample_size, seed):
    rng = np.random.RandomState(seed=seed)
    n_samples = len(predictions)
    n_size = sample_size or n_samples
    c_indices_bootstrap = []

    for bootstrap_iter in range(n_bootstraps):
        boot_indices = rng.choice(n_samples, size=n_size, replace=True)
        boot_predictions = predictions[boot_indices]
        boot_times = times[boot_indices]
        boot_events = events[boot_indices]

        try:
            c_index = concordance_index_censored(
                boot_events.astype(bool), boot_times, boot_predictions
            )[0]
            c_indices_bootstrap.append(c_index)
            logger.info(
                "Bootstrap %d/%d C-index: %.3f",
                bootstrap_iter + 1,
                n_bootstraps,
                c_index,
            )
        except Exception as exc:
            logger.warning(
                "Could not calculate C-index for bootstrap %d: %s",
                bootstrap_iter + 1,
                exc,
            )
            c_indices_bootstrap.append(float("nan"))

    c_indices_bootstrap = np.asarray(c_indices_bootstrap, dtype=np.float32)
    c_indices_bootstrap = c_indices_bootstrap[~np.isnan(c_indices_bootstrap)]

    if len(c_indices_bootstrap) == 0:
        raise ValueError("All bootstrap C-index calculations failed.")

    return {
        "c_indices": c_indices_bootstrap,
        "mean": float(np.mean(c_indices_bootstrap)),
        "std": float(np.std(c_indices_bootstrap)),
        "ci_lower": float(np.percentile(c_indices_bootstrap, 2.5)),
        "ci_upper": float(np.percentile(c_indices_bootstrap, 97.5)),
        "n_bootstraps_valid": int(len(c_indices_bootstrap)),
        "sample_size": int(n_size),
    }


def save_outputs(output_dir, tcga_ids, predictions, times, events, bootstrap_results):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "bootstrap_test_predictions.npz"
    np.savez_compressed(
        predictions_path,
        tcga_ids=np.asarray(tcga_ids, dtype=object),
        predictions=predictions,
        times=times,
        events=events,
    )

    c_indices = bootstrap_results["c_indices"]
    c_indices_path = output_dir / "bootstrap_c_indices.npy"
    np.save(c_indices_path, c_indices)

    summary = {
        "mean": bootstrap_results["mean"],
        "std": bootstrap_results["std"],
        "ci_lower": bootstrap_results["ci_lower"],
        "ci_upper": bootstrap_results["ci_upper"],
        "n_bootstraps_valid": bootstrap_results["n_bootstraps_valid"],
        "sample_size": bootstrap_results["sample_size"],
    }
    summary_path = output_dir / "bootstrap_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(10, 6))
    plt.boxplot(c_indices, vert=True)
    plt.title("Bootstrapped Concordance Indices")
    plt.xticks([1], ["Multimodal (joint fusion)"])
    plt.savefig(output_dir / "bootstrapped_cindex.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved prediction cache to %s", predictions_path)
    logger.info("Saved bootstrap C-indices to %s", c_indices_path)
    logger.info("Saved bootstrap summary to %s", summary_path)


def main():
    opt = args()
    config = ConfigManager.load_config(opt.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bootstrap_output_dir = (
        Path(config.testing.output_base_dir) / f"bootstrap_{timestamp}"
    )
    bootstrap_output_dir.mkdir(parents=True, exist_ok=True)

    logging.captureWarnings(True)
    setup_logging(
        log_dir=bootstrap_output_dir,
        log_level=config.logging.log_level,
        log_name=f"bootstrap_{timestamp}.log",
    )

    config.gpu.use_multi_gpu = False

    if not config.testing.model_path:
        raise ValueError("testing.model_path must be set in the config.")

    device, gpu_ids = setup_model_and_device(config)
    model = load_model_with_multi_gpu_support(config, device, gpu_ids)
    model.to(device)
    model.eval()

    _, test_loader = create_data_loaders(config, config.data.h5_file)

    tcga_ids, predictions, times, events = collect_test_predictions(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    baseline_c_index = concordance_index_censored(
        events.astype(bool), times, predictions
    )[0]
    logger.info("Baseline test C-index: %.3f", baseline_c_index)

    bootstrap_results = bootstrap_c_indices(
        predictions=predictions,
        times=times,
        events=events,
        n_bootstraps=opt.n_bootstraps,
        sample_size=opt.sample_size,
        seed=opt.seed,
    )

    logger.info("Bootstrap mean C-index: %.3f", bootstrap_results["mean"])
    logger.info("Bootstrap std C-index: %.3f", bootstrap_results["std"])
    logger.info(
        "Bootstrap 95%% CI: (%.3f, %.3f)",
        bootstrap_results["ci_lower"],
        bootstrap_results["ci_upper"],
    )

    save_outputs(
        output_dir=bootstrap_output_dir,
        tcga_ids=tcga_ids,
        predictions=predictions,
        times=times,
        events=events,
        bootstrap_results=bootstrap_results,
    )


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()
