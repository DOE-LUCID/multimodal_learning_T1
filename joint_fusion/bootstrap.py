from datasets import CustomDataset, HDF5Dataset
import numpy as np
import matplotlib.pyplot as plt

from model import MultimodalNetwork, OmicNetwork, print_model_summary
from sksurv.metrics import concordance_index_censored  # ADD THIS IMPORT

import torch
import torch.nn as nn
from sklearn.utils import resample

import argparse

from datasets import CustomDataset, HDF5Dataset

from tqdm import tqdm  # progress bar


def args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_base_dir",
        type=str,
        help="Path to the base directory to store results.",
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
        "--input_mapping_data_path",
        type=str,
        # default="/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/joint_fusion/",
        default="/lus/eagle/clone/g2/projects/GeomicVar/jozhw/multimodal_learning_T1/joint_fusion/",
        help="Path to input mapping data file",
    )
    parser.add_argument(
        "--input_wsi_path",
        type=str,
        default="/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_13july/tiles/256px_9.9x/combined/",
        help="Path to input WSI tiles",
    )
    # parser.add_argument('--input_wsi_embeddings_path', type=str,
    #                     default='/mnt/c/Users/tnandi/Downloads/multimodal_lucid/multimodal_lucid/early_fusion_inputs/',
    #                     help='Path to WSI embeddings generated from pretrained pathology foundation model')
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=1, help="Batch size for validation"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=1, help="Batch size for testing"
    )
    parser.add_argument(
        "--input_size_wsi", type=int, default=256, help="input_size for path images"
    )
    parser.add_argument(
        "--embedding_dim_wsi", type=int, default=384, help="embedding dimension for WSI"
    )
    parser.add_argument(
        "--embedding_dim_omic",
        type=int,
        default=256,
        help="embedding dimension for omic",
    )
    parser.add_argument(
        "--input_mode", type=str, default="wsi_omic", help="wsi, omic, wsi_omic"
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="joint",
        help="early, late, joint, joint_omic, unimodal",
    )
    parser.add_argument(
        "--calc_saliency_maps",
        action="store_true",
        help="whether to calculate saliency maps for WSI patches",
    )
    parser.add_argument(
        "--calc_IG",
        action="store_true",
        help="whether to calculate IG for RNASeq data",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model for testing.",
    )

    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",
    )
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="Use multiple GPUs if available"
    )
    parser.add_argument(
        "--joint_embedding",
        type=str,
        default="weighted_avg",
        help="Joint embedding creation method for joint fusion. Current options are concatenate, weighted_avg, and weighted_avg_dynamic",
    )
    return parser.parse_args()


def bootstrap_model(
    model,
    test_loader,
    device,
    opt,
    n_bootstraps,
    n_size,
    seed_value,
    excluded_ids=None,
    scaler_path=None,
):
    """
    Bootstrap the model to get a distribution of C-indices

    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test set
        device: torch device
        opt: options/arguments object
        n_bootstraps: number of bootstrap iterations
        seed_value: random seed
        excluded_ids: list of TCGA IDs to exclude

    Returns:
        dict containing bootstrap statistics and c_indices array
    """

    if excluded_ids is None:
        excluded_ids = ["TCGA-05-4395", "TCGA-86-8281"]

    # Load the scaler if provided
    scaler = None
    if scaler_path is not None:
        try:
            scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            print(f"Warning: Could not load scaler from {scaler_path}: {e}")
            print("Proceeding without scaling - this may cause poor performance!")

    rng = np.random.RandomState(seed=seed_value)

    # Extract the base model from DataParallel wrapper if it exists
    if isinstance(model, nn.DataParallel):
        test_model = model.module
        print("Removed DataParallel wrapper for testing")
    else:
        test_model = model

    # Move to single GPU and ensure it's in eval mode
    test_model = test_model.to(device)
    test_model.eval()

    # Phase 1: precommute
    print("\n" + "=" * 60)
    print("PHASE 1: Pre-computing embeddings and predictions...")
    print("=" * 60)

    all_predictions = []
    all_times = []
    all_events = []
    all_tcga_ids = []

    with torch.no_grad():
        for batch_idx, (
            tcga_id,
            days_to_event,
            event_occurred,
            x_wsi,
            x_omic,
        ) in enumerate(test_loader):
            # assuming that batch size is 1
            if tcga_id[0] in excluded_ids:
                print(f"Skipping TCGA ID: {tcga_id}")
                continue

            if scaler is not None:
                omic_data_np = omic_data.cpu().numpy()
                omic_data_scaled = scaler.transform(omic_data_np)
                omic_data = torch.from_numpy(omic_data_scaled).float()

            x_wsi = [tile.to(device) for tile in x_wsi]
            x_omic = x_omic.to(device)

            outputs, _, _ = test_model(opt, tcga_id, x_wsi=x_wsi, x_omic=x_omic)

            all_predictions.append(outputs.squeeze().detach().cpu().numpy())
            all_times.append(days_to_event.cpu().numpy())
            all_events.append(event_occurred.cpu().numpy())
            all_tcga_ids.append(tcga_id[0])

            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

    # transform from tensor to npy
    all_predictions = np.array(
        [np.asarray(pred).flatten()[0] for pred in all_predictions]
    )
    all_times = np.concatenate(all_times)
    all_events = np.concatenate(all_events)

    n_samples = len(all_predictions)
    print(f"\nPre-computed predictions for {n_samples} patients")
    print(
        f"Memory footprint: ~{(all_predictions.nbytes + all_times.nbytes + all_events.nbytes) / 1024**2:.2f} MB"
    )

    # dont need model anymore
    del test_model
    torch.cuda.empty_cache()

    # Phase 2: Boostrapping

    print("\n" + "=" * 60)
    print("PHASE 2: Running bootstrap iterations...")
    print("=" * 60)

    c_indices_bootstrap = []

    for bootstrap_iter in tqdm(range(n_bootstraps), desc="Bootstrap iteration"):

        # Resample indices with replacement
        boot_indices = rng.choice(n_samples, size=n_size, replace=True)

        # diagnostics
        if bootstrap_iter == 0:
            unique_indices = np.unique(boot_indices)
            print(f"\n  Bootstrap diagnostic (iteration 1):")
            print(f"    - Original samples: {n_samples}")
            print(f"    - Unique samples in bootstrap: {len(unique_indices)}")
            print(
                f"    - Samples appearing multiple times: {n_samples - len(unique_indices)}"
            )
            print(f"    - Max repetitions: {np.max(np.bincount(boot_indices))}")

        boot_predictions = all_predictions[boot_indices]
        boot_times = all_times[boot_indices]
        boot_events = all_events[boot_indices]

        # Calculate CI for this bootstrap sample
        try:
            c_index = concordance_index_censored(
                boot_events.astype(bool), boot_times, boot_predictions
            )[0]
            c_indices_bootstrap.append(c_index)

            print(f"Bootstrap {bootstrap_iter + 1} C-index: {c_index:.3f}")

        except Exception as e:
            print(f"Could not calculate CI for bootstrap {bootstrap_iter + 1}: {e}")
            c_indices_bootstrap.append(float("nan"))

    # Remove any NaN values
    c_indices_bootstrap = np.array(c_indices_bootstrap)
    c_indices_bootstrap = c_indices_bootstrap[~np.isnan(c_indices_bootstrap)]

    # Calculate statistics
    mean_c_index = np.mean(c_indices_bootstrap)
    std_c_index = np.std(c_indices_bootstrap)
    ci_lower = np.percentile(c_indices_bootstrap, 2.5)
    ci_upper = np.percentile(c_indices_bootstrap, 97.5)

    print("\n" + "=" * 50)
    print("Bootstrap Results:")
    print(f"Mean C-index: {mean_c_index:.3f}")
    print(f"Standard Deviation of C-index: {std_c_index:.3f}")
    print(f"95% Confidence Interval: ({ci_lower:.3f}, {ci_upper:.3f})")
    print("=" * 50)

    # plot c indices across bootstrapped samples
    plt.figure(figsize=(10, 6))
    plt.boxplot(c_indices_bootstrap, vert=True)
    plt.title("Bootstrap Distribution of C-index")
    plt.xlabel("C-index")
    plt.ylim(0.4, 0.8)
    plt.savefig(f"bootstrapped_cindex.png", dpi=300, bbox_inches="tight")

    return {
        "c_indices": c_indices_bootstrap,
        "mean": mean_c_index,
        "std": std_c_index,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


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


def setup_model_and_device(opt):
    """Setup model and device configuration"""

    # Parse GPU IDs
    if opt.gpu_ids == "-1":
        device = torch.device("cpu")
        gpu_ids = []
        print("Using CPU")
    else:
        gpu_ids = [int(x) for x in opt.gpu_ids.split(",") if x.strip()]
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


def load_model_with_multi_gpu_support(opt, device, gpu_ids):
    """Load model with proper multi-GPU support"""

    # Create base model
    model = MultimodalNetwork(
        embedding_dim_wsi=opt.embedding_dim_wsi,
        embedding_dim_omic=opt.embedding_dim_omic,
        mode=opt.input_mode,
        fusion_type=opt.fusion_type,
        joint_embedding_type=opt.joint_embedding,
        mlp_layers=opt.mlp_layers,
        dropout=opt.dropout,
    )

    # Load checkpoint
    print(f"Loading model from: {opt.model_path}")
    checkpoint = torch.load(opt.model_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # Determine if the saved model was using DataParallel
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    use_multi_gpu = len(gpu_ids) > 1 and opt.use_multi_gpu

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
        print(f"Wrapping model with DataParallel on GPUs: {gpu_ids}")
        # model = nn.DataParallel(model, device_ids=gpu_ids)
        # Set the primary GPU
        torch.cuda.set_device(gpu_ids[0])

    # Load the state dict
    model.load_state_dict(state_dict)
    print("Model loaded successfully")

    return model


if __name__ == "__main__":
    opt = args()

    n_bootstraps = 100
    n_size = 1000
    seed_value = 422

    device, gpu_ids = setup_model_and_device(opt)
    model = load_model_with_multi_gpu_support(opt, device, gpu_ids)

    checkpoint_dir = opt.model_path.rsplit("/", 1)[0]

    import re

    fold_match = re.search(r"fold_(\d+)", opt.model_path)
    if fold_match:
        fold_idx = int(fold_match.group(1))
        scaler_path = f"{checkpoint_dir}/scaler_fold_{fold_idx}.save"
        print(f"Using scaler from fold {fold_idx}: {scaler_path}")
    else:
        print("WARNING: Could not determine fold number from model path")
        print("You may need to specify the scaler path manually")
        scaler_path = None

    # REMOVE THESE TWO LINES - model is already loaded and on device:
    # model.to(device)
    # model = load_model_state_dict(model, opt.model_path)

    test_dataset = HDF5Dataset(
        opt,
        opt.input_mapping_data_path + "mapping_data.h5",
        split="test",
        mode=opt.input_mode,
        train_val_test="test",
    )

    # IMPORTANT: Also load validation set and combine if that's what was done during training
    val_dataset = HDF5Dataset(
        opt,
        opt.input_mapping_data_path + "mapping_data.h5",
        split="val",
        mode=opt.input_mode,
        train_val_test="val",
    )

    from torch.utils.data import ConcatDataset

    combined_test_dataset = ConcatDataset([val_dataset, test_dataset])

    test_loader = torch.utils.data.DataLoader(
        dataset=combined_test_dataset,  # Use combined dataset
        batch_size=opt.test_batch_size,
        shuffle=False,  # MUST be False for reproducibility
        num_workers=0,
        pin_memory=True,
    )

    bootstrap_results = bootstrap_model(
        model=model,
        test_loader=test_loader,
        device=device,
        opt=opt,
        n_bootstraps=n_bootstraps,
        n_size=n_size,
        seed_value=seed_value,
        excluded_ids=["TCGA-05-4395", "TCGA-86-8281"],
    )
    print(f"\nFinal Results:")
    print(
        f"Mean C-index: {bootstrap_results['mean']:.3f} ± {bootstrap_results['std']:.3f}"
    )
    print(
        f"95% CI: ({bootstrap_results['ci_lower']:.3f}, {bootstrap_results['ci_upper']:.3f})"
    )
