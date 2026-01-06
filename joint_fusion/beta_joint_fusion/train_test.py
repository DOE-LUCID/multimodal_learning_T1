import random
import math
import gc
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import pandas as pd
import typing as tp
import torch
import os
import wandb
import time
import argparse
import json
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader, Subset
import torch.utils.checkpoint as checkpoint
from torch.optim.lr_scheduler import _LRScheduler

# import torch.optim.lr_scheduler as lr_scheduler
from loss_function import JointLoss
from torchsummary import summary
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from utils.utils import mixed_collate, clear_memory, monitor_memory
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from datasets import CustomDataset, HDF5Dataset
from models import MultimodalNetwork, OmicNetwork, print_model_summary
from sklearn.model_selection import StratifiedKFold
from generate_wsi_embeddings import CustomDatasetWSI

import h5py

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pdb
import pickle
import os
from pdb import set_trace
from captum.attr import IntegratedGradients, Saliency

import torchviz


torch.autograd.set_detect_anomaly(True)

current_time = datetime.now().strftime("%y_%m_%d_%H_%M")


class StratifiedBatchSampler:
    """
    Combines time stratification with risk-set preservation
    """

    def __init__(self, times, events, batch_size, min_risk_coverage=0.8, shuffle=True):
        self.times = np.array(times)
        self.events = np.array(events)
        self.batch_size = batch_size
        self.min_risk_coverage = min_risk_coverage
        self.shuffle = shuffle

        self.indices = np.arange(len(times))

        # Create time strata for balanced representation
        self.time_quantiles = np.quantile(times[events == 1], [0, 0.33, 0.67, 1.0])
        self.event_time_strata = (
            np.digitize(times[events == 1], self.time_quantiles) - 1
        )

    def __iter__(self):

        # Get all available indices
        available_indices = set(self.indices.copy())

        if self.shuffle:
            available_indices = set(np.random.permutation(list(available_indicies)))

        # Generate baches until out of samples
        while len(available_indices) >= self.batch_size:
            batch = self._create_hybrid_batch(available_indices)

            # Remove used indices from available set
            for idx in batch:
                available_indices.discard(idx)

            yield batch

        # Drop incomplete batches so no need to handle leftover indices

    def __len__(self):
        """Return number of batches per epoch"""
        return len(self.indices) // self.batch_size

    def _create_hybrid_batch(self, available_indices):
        """Create batch with both time balance and risk-set preservation"""
        batch = []
        available_list = list(available_indices)

        # Step 1: Select events with temporal diversity
        event_candidates = [i for i in available_list if self.events[i] == 1]

        if event_candidates:
            # Group events by time strata
            early_events = [
                i for i in event_candidates if self.times[i] <= self.time_quantiles[1]
            ]
            mid_events = [
                i
                for i in event_candidates
                if self.time_quantiles[1] < self.times[i] <= self.time_quantiles[2]
            ]
            late_events = [
                i for i in event_candidates if self.times[i] > self.time_quantiles[2]
            ]

            # Sample events from each stratum (temporal balance)
            max_events_per_stratum = max(1, self.batch_size // 6)  # Conservative
            selected_events = []

            for event_group in [early_events, mid_events, late_events]:
                if event_group:
                    n_select = min(len(event_group), max_events_per_stratum)
                    selected = np.random.choice(event_group, n_select, replace=False)
                    selected_events.extend(selected)

            batch.extend(selected_events)

            # Step 2: For each selected event, ensure risk-set coverage
            for event_idx in selected_events:
                event_time = self.times[event_idx]

                # Find at-risk samples
                at_risk_candidates = [
                    i
                    for i in available_list
                    if i not in batch and self.times[i] >= event_time
                ]

                if at_risk_candidates:
                    # Calculate needed coverage
                    total_at_risk = np.sum(self.times >= event_time)
                    needed = int(self.min_risk_coverage * total_at_risk)
                    needed = min(needed, len(at_risk_candidates))

                    # Sample at-risk with some temporal diversity
                    at_risk_candidates.sort(key=lambda i: self.times[i])

                    # Use stratified sampling within at-risk set
                    if needed > 1:
                        indices = np.linspace(
                            0, len(at_risk_candidates) - 1, needed
                        ).astype(int)
                        selected_at_risk = [at_risk_candidates[j] for j in indices]
                    else:
                        selected_at_risk = at_risk_candidates[:needed]

                    # Add to batch if space allows
                    space_remaining = self.batch_size - len(batch)
                    batch.extend(selected_at_risk[:space_remaining])

                    if len(batch) >= self.batch_size:
                        break

        # Step 3: Fill remaining slots with temporal balance
        remaining_slots = self.batch_size - len(batch)
        if remaining_slots > 0:
            remaining_candidates = [i for i in available_list if i not in batch]

            if remaining_candidates:
                # Add remaining samples with time diversity preference
                remaining_times = [self.times[i] for i in remaining_candidates]
                time_weights = self._compute_time_diversity_weights(
                    remaining_times, batch
                )

                # Weighted sampling for time diversity
                probs = np.array(time_weights)
                probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs)

                n_additional = min(remaining_slots, len(remaining_candidates))
                additional = np.random.choice(
                    remaining_candidates, size=n_additional, replace=False, p=probs
                )
                batch.extend(additional)

        return batch[: self.batch_size]

    def _compute_time_diversity_weights(self, candidate_times, current_batch):
        """Compute weights favoring temporal diversity"""
        if not current_batch:
            return [1.0] * len(candidate_times)

        batch_times = [self.times[i] for i in current_batch]
        batch_time_range = (min(batch_times), max(batch_times))

        weights = []
        for t in candidate_times:
            # Higher weight for times that expand the temporal range
            if t < batch_time_range[0] or t > batch_time_range[1]:
                weights.append(2.0)  # Expand range
            else:
                weights.append(1.0)  # Within range

        return weights


def create_data_loaders(opt, h5_file):
    full_dataset = HDF5Dataset(
        opt, h5_file, split="train", mode=opt.input_mode, train_val_test="train"
    )

    times, events = extract_survival_data(full_dataset)

    batch_sampler = StratifiedBatchSampler(
        times=times,
        events=events,
        batch_size=opt.batch_size,
        min_risk_coverage=0.8,
        shuffle=True,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=full_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,  # 8,
        # prefetch_factor=2,
        pin_memory=True,
    )

    validation_loader = torch.utils.data.DataLoader(
        dataset=HDF5Dataset(
            opt, h5_file, split="val", mode=opt.input_mode, train_val_test="val"
        ),
        batch_size=opt.val_batch_size,
        shuffle=True,
        num_workers=0,  # 8,
        # prefetch_factor=2,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=HDF5Dataset(
            opt, h5_file, split="test", mode=opt.input_mode, train_val_test="test"
        ),
        batch_size=opt.test_batch_size,
        shuffle=True,
        num_workers=0,  # 1, #4,
        # prefetch_factor=2,
        pin_memory=True,
    )

    return train_loader, validation_loader, test_loader


def print_gpu_memory_usage():
    """Print detailed GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")


def plot_survival_distributions(
    times_train: np.ndarray,
    events_train: np.ndarray,
    times_val: np.ndarray,
    events_val: np.ndarray,
    fold_idx: int,
    save_dir: str = "./fold_distributions",
) -> None:
    """
    Plot survival time distributions for train and validation sets.

    Args:
        times_train: Training survival times
        events_train: Training event indicators
        times_val: Validation survival times
        events_val: Validation event indicators
        fold_idx: Current fold index
        save_dir: Directory to save plots
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Fold {fold_idx + 1} - Survival Data Distributions", fontsize=16)

    # 1. Survival times distribution
    axes[0, 0].hist(
        times_train, bins=30, alpha=0.7, label="Train", color="blue", density=True
    )
    axes[0, 0].hist(
        times_val, bins=30, alpha=0.7, label="Validation", color="red", density=True
    )
    axes[0, 0].set_xlabel("Survival Time (days)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Survival Times Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Event rates
    train_event_rate = events_train.mean()
    val_event_rate = events_val.mean()

    axes[0, 1].bar(
        ["Train", "Validation"],
        [train_event_rate, val_event_rate],
        color=["blue", "red"],
        alpha=0.7,
    )
    axes[0, 1].set_ylabel("Event Rate")
    axes[0, 1].set_title("Event Rates Comparison")
    axes[0, 1].set_ylim(0, 1)

    # Add text annotations
    axes[0, 1].text(
        0, train_event_rate + 0.02, f"{train_event_rate:.3f}", ha="center", va="bottom"
    )
    axes[0, 1].text(
        1, val_event_rate + 0.02, f"{val_event_rate:.3f}", ha="center", va="bottom"
    )

    # 3. Box plots for survival times by event status
    train_data = pd.DataFrame(
        {"times": times_train, "events": events_train, "split": "Train"}
    )
    val_data = pd.DataFrame(
        {"times": times_val, "events": events_val, "split": "Validation"}
    )

    combined_data = pd.concat([train_data, val_data])
    combined_data["event_status"] = combined_data["events"].map(
        {0: "Censored", 1: "Event"}
    )

    sns.boxplot(
        data=combined_data, x="split", y="times", hue="event_status", ax=axes[0, 2]
    )
    axes[0, 2].set_title("Survival Times by Event Status")
    axes[0, 2].set_ylabel("Survival Time (days)")

    # 4. Cumulative distribution of survival times
    train_sorted = np.sort(times_train)
    val_sorted = np.sort(times_val)

    train_cumprob = np.arange(1, len(train_sorted) + 1) / len(train_sorted)
    val_cumprob = np.arange(1, len(val_sorted) + 1) / len(val_sorted)

    axes[1, 0].plot(
        train_sorted, train_cumprob, label="Train", color="blue", linewidth=2
    )
    axes[1, 0].plot(
        val_sorted, val_cumprob, label="Validation", color="red", linewidth=2
    )
    axes[1, 0].set_xlabel("Survival Time (days)")
    axes[1, 0].set_ylabel("Cumulative Probability")
    axes[1, 0].set_title("Cumulative Distribution of Survival Times")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Sample size information
    train_n = len(times_train)
    val_n = len(times_val)
    train_events = events_train.sum()
    val_events = events_val.sum()

    info_text = f"""
    Training Set:
    • Total samples: {train_n}
    • Events: {train_events} ({train_event_rate:.1%})
    • Censored: {train_n - train_events} ({1-train_event_rate:.1%})
    
    Validation Set:
    • Total samples: {val_n}
    • Events: {val_events} ({val_event_rate:.1%})
    • Censored: {val_n - val_events} ({1-val_event_rate:.1%})
    
    Ratio (Val/Train): {val_n/train_n:.3f}
    """

    axes[1, 1].text(
        0.1,
        0.9,
        info_text,
        transform=axes[1, 1].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )
    axes[1, 1].axis("off")

    # 6. Time bins distribution (stratification groups)
    strat_labels_train = _create_stratification_labels(times_train, events_train)
    strat_labels_val = _create_stratification_labels(times_val, events_val)

    unique_train, counts_train = np.unique(strat_labels_train, return_counts=True)
    unique_val, counts_val = np.unique(strat_labels_val, return_counts=True)

    # Ensure both have same groups for comparison
    all_groups = np.union1d(unique_train, unique_val)
    train_counts_full = np.array(
        [
            counts_train[np.where(unique_train == g)[0][0]] if g in unique_train else 0
            for g in all_groups
        ]
    )
    val_counts_full = np.array(
        [
            counts_val[np.where(unique_val == g)[0][0]] if g in unique_val else 0
            for g in all_groups
        ]
    )

    x_pos = np.arange(len(all_groups))
    width = 0.35

    axes[1, 2].bar(
        x_pos - width / 2,
        train_counts_full,
        width,
        label="Train",
        color="blue",
        alpha=0.7,
    )
    axes[1, 2].bar(
        x_pos + width / 2,
        val_counts_full,
        width,
        label="Validation",
        color="red",
        alpha=0.7,
    )
    axes[1, 2].set_xlabel("Stratification Group")
    axes[1, 2].set_ylabel("Sample Count")
    axes[1, 2].set_title("Sample Distribution Across Stratification Groups")
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels([f"G{g}" for g in all_groups])
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(save_dir, f"fold_{fold_idx + 1}_distributions.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Distribution plots saved to: {save_path}")


def _create_stratification_labels(
    times: np.ndarray,
    events: np.ndarray,
    n_time_bins: int = 5,
    strategy: str = "quantile",
) -> np.ndarray:
    """
    Create stratification labels for survival data by combining time bins and event status.
    """
    print(f"Creating {n_time_bins} time bins using {strategy} strategy")

    # Handle edge cases
    if len(times) != len(events):
        raise ValueError("Times and events arrays must have the same length")

    if len(times) == 0:
        raise ValueError("Empty arrays provided")

    # Create time bins
    try:
        # Adjust n_time_bins if we have too few unique times
        unique_times = len(np.unique(times))
        if unique_times < n_time_bins:
            print(
                f"Warning: Only {unique_times} unique times, reducing time bins to {unique_times}"
            )
            n_time_bins = unique_times

        discretizer = KBinsDiscretizer(
            n_bins=n_time_bins, encode="ordinal", strategy=strategy, subsample=None
        )

        # Fit and transform times
        time_bins = (
            discretizer.fit_transform(times.reshape(-1, 1)).flatten().astype(int)
        )

        print(f"Time bin distribution:")
        unique_bins, bin_counts = np.unique(time_bins, return_counts=True)
        for bin_idx, count in zip(unique_bins, bin_counts):
            print(f"  Time bin {bin_idx}: {count} samples")

    except Exception as e:
        print(f"Error in time binning: {e}")
        # Fallback: create simple quantile-based bins manually
        quantiles = np.linspace(0, 1, n_time_bins + 1)
        bin_edges = np.quantile(times, quantiles)
        # Ensure no duplicate edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) <= 1:
            # If all times are the same, create a single bin
            time_bins = np.zeros(len(times), dtype=int)
        else:
            time_bins = np.digitize(
                times, bin_edges[1:]
            )  # Use right edges, so bin 0 is first bin

        print(f"Fallback binning created {len(np.unique(time_bins))} bins")

    # Create combined stratification labels: time_bin * 2 + event_status
    # This creates up to 2 * n_time_bins unique labels
    strat_labels = time_bins * 2 + events.astype(int)

    print(f"Created stratification labels:")
    print(f"  - {len(np.unique(strat_labels))} unique stratification groups")
    print(f"  - Range: {strat_labels.min()} to {strat_labels.max()}")

    return strat_labels


def create_stratified_survival_folds(
    times: np.ndarray,
    events: np.ndarray,
    n_splits: int = 5,
    n_time_bins: int = 5,
    random_state: int = 42,
    min_samples_per_group: int = 2,
    strategy: str = "quantile",
) -> tp.List[tp.Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified K-fold splits for survival data.

    Args:
        times: Array of survival times
        events: Array of event indicators
        n_splits: Number of folds
        n_time_bins: Number of time bins for stratification
        random_state: Random seed
        min_samples_per_group: Minimum samples required per stratification group
        strategy: Binning strategy for time discretization

    Returns:
        List of (train_indices, val_indices) tuples
    """
    print(f"Creating stratified folds with {n_splits} splits, {n_time_bins} time bins")
    print(
        f"Input data: {len(times)} samples, {events.sum()} events ({events.mean():.1%})"
    )

    # Validate inputs
    if len(times) != len(events):
        raise ValueError("Times and events arrays must have the same length")

    if len(times) < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} folds with only {len(times)} samples"
        )

    # Create stratification labels
    try:
        strat_labels = _create_stratification_labels(
            times, events, n_time_bins, strategy
        )
        print(
            f"Created stratification labels with {len(np.unique(strat_labels))} unique groups"
        )

        # Print group distribution
        unique_labels, counts = np.unique(strat_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            time_bin = label // 2
            event_status = label % 2
            print(
                f"  Group {label}: time_bin={time_bin}, event={event_status}, count={count}"
            )

    except Exception as e:
        print(f"Error creating stratification labels: {e}")
        raise

    # Check group sizes and merge small groups
    unique_labels, counts = np.unique(strat_labels, return_counts=True)
    small_groups = unique_labels[counts < min_samples_per_group]

    if len(small_groups) > 0:
        print(
            f"Warning: Found {len(small_groups)} groups with < {min_samples_per_group} samples"
        )
        print(
            f"Small groups: {small_groups} with counts: {counts[np.isin(unique_labels, small_groups)]}"
        )

        # Merge small groups with the nearest group
        for small_group in small_groups:
            # Find the nearest group by label value
            other_groups = unique_labels[unique_labels != small_group]
            if len(other_groups) > 0:
                nearest_group = other_groups[
                    np.argmin(np.abs(other_groups - small_group))
                ]
                print(f"  Merging group {small_group} into group {nearest_group}")
                strat_labels[strat_labels == small_group] = nearest_group
            else:
                print(f"  Warning: No other groups to merge group {small_group} into")

    # Check final group distribution
    final_unique_labels, final_counts = np.unique(strat_labels, return_counts=True)
    print(f"Final stratification: {len(final_unique_labels)} groups")
    for label, count in zip(final_unique_labels, final_counts):
        print(f"  Group {label}: {count} samples")

    # Ensure we have enough groups for stratification
    if len(final_unique_labels) < 2:
        print("Warning: Too few stratification groups, falling back to regular K-fold")
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(kf.split(np.arange(len(times))))

    # Create stratified folds
    try:
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

        # Generate dummy X for stratification (we only care about the stratification labels)
        dummy_X = np.arange(len(times)).reshape(-1, 1)

        # Generate folds
        folds = list(skf.split(dummy_X, strat_labels))

        print(f"Successfully created {len(folds)} stratified folds")

        # Verify fold quality
        for i, (train_idx, val_idx) in enumerate(folds):
            train_event_rate = events[train_idx].mean()
            val_event_rate = events[val_idx].mean()
            train_median_time = np.median(times[train_idx])
            val_median_time = np.median(times[val_idx])

            print(
                f"  Fold {i+1}: Train({len(train_idx)}) - events={train_event_rate:.3f}, "
                f"median_time={train_median_time:.0f} | "
                f"Val({len(val_idx)}) - events={val_event_rate:.3f}, "
                f"median_time={val_median_time:.0f}"
            )

        return folds

    except Exception as e:
        print(f"Error in StratifiedKFold: {e}")
        print("Falling back to regular K-fold")
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(kf.split(np.arange(len(times))))


def extract_survival_data(dataset) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Extract survival times and events from a dataset.

    Args:
        dataset: PyTorch dataset containing survival data

    Returns:
        Tuple of (times, events) arrays
    """
    times = []
    events = []

    print(f"Extracting survival data from {len(dataset)} samples...")

    for i in range(len(dataset)):
        try:
            # Assuming dataset returns (tcga_id, days_to_event, event_occurred, x_wsi, x_omic)
            sample = dataset[i]

            if len(sample) < 3:
                print(
                    f"Warning: Sample {i} has unexpected format: {len(sample)} elements"
                )
                continue

            times.append(float(sample[1]))  # days_to_event
            events.append(int(sample[2]))  # event_occurred

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    if len(times) == 0:
        raise ValueError("No valid survival data found in dataset")

    times_array = np.array(times)
    events_array = np.array(events)

    print(f"Extracted {len(times_array)} valid samples")
    print(f"Times range: {times_array.min():.1f} - {times_array.max():.1f}")
    print(
        f"Events: {events_array.sum()} out of {len(events_array)} ({events_array.mean():.3f})"
    )

    return times_array, events_array


class CosineAnnealingWarmRestartsDecay(_LRScheduler):
    def __init__(
        self,
        optimizer,
        T_0,
        T_mult=1,
        eta_min=0,
        decay_factor=1.0,
        last_epoch=-1,
        verbose=False,
    ):
        """
        Cosine annealing with warm restarts and optional learning rate decay between cycles.

        Args:
            optimizer: Wrapped optimizer
            T_0: Number of iterations for the first restart
            T_mult: A factor increases T_i after a restart
            eta_min: Minimum learning rate
            decay_factor: Factor by which max_lr is reduced after each cycle (1.0 = no decay)
            last_epoch: The index of last epoch
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay_factor = decay_factor
        self.T_i = T_0  # Current period length
        self.T_cur = 0  # Current step within period
        self.cycle = 0  # Current cycle number

        # Store initial learning rates for each param group
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_max_lrs = self.base_lrs.copy()

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):

        return [
            self.eta_min
            + (max_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for max_lr in self.current_max_lrs
        ]

    def step(self, epoch=None):
        """
        1. First calculates and applies the current learning rate
        2. Updates the internal state for the next step
        3. Handles cycle transitions after the current step is complete
        """

        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

        self.T_cur += 1

        if self.T_cur > self.T_i:
            # Cycle completed, prep for next one
            self.cycle += 1

            self.current_max_lrs = [
                lr * self.decay_factor for lr in self.current_max_lrs
            ]

            # Restart
            self.T_cur = 0
            self.T_i = int(self.T_i * self.T_mult)


def train_nn(opt, h5_file, device, plot_distributions=True):

    from pathlib import Path
    from dotenv import load_dotenv

    env_path: Path = Path.cwd() / ".env"

    load_dotenv(env_path)

    project = os.getenv("WANDB_PROJECT")
    entity = os.getenv("WANDB_ENTITY")

    wandb.init(
        project=project,
        entity=entity,
        config={
            "num_folds": opt.n_folds,
            "num_epochs": opt.num_epochs,
            "batch_size": opt.batch_size,
            "val_batch_size": opt.val_batch_size,
            "lr": opt.lr,
            "mlp_layers": opt.mlp_layers,
            "dropout": opt.dropout,
            "fusion_type": opt.fusion_type,
            "joint_embedding": opt.joint_embedding,
            "embedding_dim_wsi": opt.embedding_dim_wsi,
            "embedding_dim_omic": opt.embedding_dim_omic,
            "input_mode": opt.input_mode,
            "stratified_cv": True,
            "use_pretrained_omic": opt.use_pretrained_omic,
            "omic_checkpoint_path": (
                opt.omic_checkpoint_path if opt.use_pretrained_omic else None
            ),
        },
    )

    if opt.scheduler == "cosine_warmer":

        wandb.config.update(
            {
                "scheduler": opt.scheduler,
                "scheduler_params": {
                    "t_0": opt.t_0,
                    "t_mult": opt.t_mult,
                    "eta_min": opt.eta_min,
                },
            },
        )

    elif opt.scheduler == "exponential":

        wandb.config.update(
            {
                "scheduler": opt.scheduler,
                "scheduler_params": {
                    "exp_gamma": opt.exp_gamma,
                },
            },
        )
    elif opt.scheduler == "step_lr":

        wandb.config.update(
            {
                "scheduler": opt.scheduler,
                "scheduler_params": {"exp_gamma": opt.exp_gamma, "step": opt.lr_step},
            }
        )
    current_time = datetime.now()

    # If user provided timestamp then use for consistency
    if opt.timestamp:

        checkpoint_dir = "checkpoints/checkpoint_" + opt.timestamp
        os.makedirs(checkpoint_dir, exist_ok=True)

    else:

        checkpoint_dir = "checkpoints/checkpoint_" + current_time.strftime(
            "%Y-%m-%d-%H-%M-%S"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader, _, test_loader = create_data_loaders(opt, h5_file)
    dataset = (
        train_loader.dataset
    )  # use only the training dataset for CV during training [both the validation and test splits can be used for testing]

    total_samples = len(dataset)
    print(f"total training data size: {total_samples} samples")

    # Extract survival data for stratification
    print("Extracting survival data for startified CV...")
    times, events = extract_survival_data(dataset)

    print(f"Survival data summary:")
    print(f"  - Total samples: {len(times)}")
    print(f"  - Events: {events.sum()} ({events.mean():.1%})")
    print(f"  - Censored: {len(events) - events.sum()} ({1 - events.mean():.1%})")
    print(f"  - Median survival time: {np.median(times):.1f} days")
    print(f"  - Time range: {times.min():.1f} - {times.max():.1f} days")

    try:
        folds = create_stratified_survival_folds(
            times=times,
            events=events,
            n_splits=opt.n_folds,
            n_time_bins=10,
        )
        print(f"Successfully created {len(folds)} stratified folds")

    except Exception as e:
        print(f"Error creating stratified folds: {e}")
        # Fallback to regular k-fold if stratified fails
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=opt.n_folds, shuffle=True, random_state=6)
        folds = list(kf.split(np.arange(len(dataset))))
        print("Falling back to regular K-Fold CV")

    num_folds = opt.n_folds
    num_epochs = opt.num_epochs

    # Obtain Validation CI and Loss avg at each epoch
    ci_average_by_epoch = np.zeros(opt.num_epochs, dtype=np.float32)
    loss_average_by_epoch = np.zeros(opt.num_epochs, dtype=np.float32)

    # need to keep fold index at 0 for the sake of the averaging by epoch
    if opt.use_mixed_precision:
        amp_scaler = GradScaler()
        print("Using mixed precision training with StandardScaler")

    # Step counter for wandb
    step_counter = 0
    # Main fold iteration loop
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{len(folds)}")
        print(f"{'='*60}")

        # Print fold statistics
        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")

        # Extract survival data for this fold to verify stratification
        times_train_fold = times[train_idx]
        events_train_fold = events[train_idx]
        times_val_fold = times[val_idx]
        events_val_fold = events[val_idx]

        train_event_rate = events_train_fold.mean()
        val_event_rate = events_val_fold.mean()

        print(f"Train event rate: {train_event_rate:.3f}")
        print(f"Validation event rate: {val_event_rate:.3f}")
        print(f"Train median time: {np.median(times_train_fold):.1f} days")
        print(f"Validation median time: {np.median(times_val_fold):.1f} days")

        # Plot distributions if requested
        if plot_distributions:
            plot_survival_distributions(
                times_train_fold,
                events_train_fold,
                times_val_fold,
                events_val_fold,
                fold_idx,
                save_dir=os.path.join(checkpoint_dir, "fold_distributions"),
            )

        print_gpu_memory_usage()  # Monitor memory at start of each fold

        # Create subsets for this fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create data loaders for this fold
        train_loader_fold = DataLoader(
            train_subset,
            batch_size=opt.batch_size,
            collate_fn=mixed_collate,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
        val_loader_fold = DataLoader(
            val_subset,
            batch_size=opt.val_batch_size,
            collate_fn=mixed_collate,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Compute number of batches per epoch
        num_batches_per_epoch = len(train_loader_fold)
        print(
            f"Number of batches per epoch for fold {fold_idx + 1}: {num_batches_per_epoch}"
        )

        # Update wandb config
        wandb.config.update(
            {f"num_batches_per_epoch_fold_{fold_idx}": num_batches_per_epoch},
            allow_val_change=True,
        )

        model = MultimodalNetwork(
            embedding_dim_wsi=opt.embedding_dim_wsi,
            embedding_dim_omic=opt.embedding_dim_omic,
            mode=opt.input_mode,
            fusion_type=opt.fusion_type,
            joint_embedding_type=opt.joint_embedding,
            mlp_layers=opt.mlp_layers,
            dropout=opt.dropout,
        )

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
            torch.cuda.set_device(0)

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
        if opt.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=opt.exp_gamma
            )
        elif opt.scheduler == "cosine_warmer":
            scheduler = CosineAnnealingWarmRestartsDecay(
                optimizer,
                T_0=opt.t_0,
                T_mult=opt.t_mult,
                eta_min=opt.eta_min,
                decay_factor=opt.decay_factor,
            )

        elif opt.scheduler == "step_lr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=opt.step_lr_step, gamma=opt.step_lr_gamma
            )
        else:
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )

        joint_loss = JointLoss()

        # Initialize and fit the scaler for this fold
        print(f"Fitting scaler for fold {fold_idx + 1}...")
        scaler = StandardScaler()
        for batch_idx, (tcga_id, _, _, _, x_omic) in enumerate(train_loader_fold):
            if batch_idx % 10 == 0:  # Print progress every 10 batches
                print(f"  Fitting batch {batch_idx + 1}/{len(train_loader_fold)}")
            x_omic_np = x_omic.cpu().numpy()
            scaler.partial_fit(x_omic_np)

        # Save the scaler for the current fold
        scaler_path = os.path.join(checkpoint_dir, f"scaler_fold_{fold_idx}.save")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved for fold {fold_idx + 1} at {scaler_path}")

        # EPOCH TRAINING LOOP for this fold
        for epoch in range(opt.num_epochs):
            print(
                f"\n--- Fold {fold_idx + 1}/{len(folds)}, Epoch {epoch + 1}/{opt.num_epochs} ---"
            )
            start_train_time = time.time()
            model.train()

            # # added to reduce memory requirement
            # if isinstance(model, torch.nn.DataParallel):
            #     model.module.gradient_checkpointing_enable()
            # else:
            #     model.gradient_checkpointing_enable()

            loss_epoch = 0

            # Add debug info for first batch
            print(f"Starting epoch {epoch}, about to iterate over training batches...")
            print(f"Training loader has {len(train_loader_fold)} batches")
            # log the learning rate at the start of the epoch
            current_lr = optimizer.param_groups[0]["lr"]

            # model training in batches for the train dataloader for the current fold
            for batch_idx, (
                tcga_id,
                days_to_event,
                event_occurred,
                x_wsi,
                x_omic,
            ) in enumerate(train_loader_fold):
                # x_wsi is a list of tensors (one tensor for each tile)
                print(
                    f"Total training samples in fold: {len(train_loader_fold.dataset)}"
                )
                print(f"Batch size: {opt.batch_size}")
                print(f"Batch index: {batch_idx} out of {num_batches_per_epoch}")

                # print(f"Before scaling - x_omic shape: {x_omic.shape}")
                # print(f"Before scaling - days_to_event shape: {days_to_event.shape}")
                # print(f"Before scaling - event_occurred shape: {event_occurred.shape}")

                # NOTE: Do not apply standard scaler to omic data
                x_wsi = [x.to(device) for x in x_wsi]
                x_omic = x_omic.to(device)

                print(f"After scaling - x_omic shape: {x_omic.shape}")
                days_to_event = days_to_event.to(device)
                # days_to_last_followup = days_to_last_followup.to(device)
                event_occurred = event_occurred.to(device)

                # print(f"Final - days_to_event shape: {days_to_event.shape}")
                # print(f"Final - event_occurred shape: {event_occurred.shape}")
                print("Days to event: ", days_to_event)
                print("event occurred: ", event_occurred)

                optimizer.zero_grad()

                if opt.use_mixed_precision:
                    with autocast():  # should wrap only the forward pass including the loss calculation
                        predictions, wsi_embedding, omic_embedding = model(
                            opt,
                            tcga_id,
                            x_wsi=x_wsi,
                            x_omic=x_omic,  # Now properly scaled
                        )

                        # print(f"Model predictions shape: {predictions.shape}")
                        loss = joint_loss(
                            predictions.squeeze(),
                            days_to_event,
                            event_occurred,
                            wsi_embedding,
                            omic_embedding,
                        )
                        print("\n loss (train, mixed precision): ", loss.data.item())
                        loss_epoch += loss.data.item()

                    # Mixed precision backward pass
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    print(" Not using mixed precision")
                    # model for survival outcome (uses Cox PH partial log likelihood as the loss function)
                    # the model output should be considered as beta*X to be used in the Cox loss function

                    start_time = time.time()

                    predictions, wsi_embedding, omic_embedding = model(
                        opt,
                        tcga_id,
                        x_wsi=x_wsi,  # list of tensors (one for each tile)
                        x_omic=x_omic,
                    )
                    # print(f"predictions: {predictions} from train_test.py")
                    step1_time = time.time()
                    loss = joint_loss(
                        predictions.squeeze(),
                        # predictions are not survival outcomes, rather log-risk scores beta*X
                        days_to_event,
                        event_occurred,
                        wsi_embedding,
                        omic_embedding,
                    )  # Cox partial likelihood loss for survival outcome prediction
                    print("\n loss (train): ", loss.data.item())
                    step2_time = time.time()
                    loss_epoch += (
                        loss.data.item()
                    )  # * len(tcga_id)  # multiplying loss by batch size for accurate epoch averaging
                    # backpropagate loss through the entire model arch upto the inputs
                    loss.backward(
                        retain_graph=True if epoch == 0 and batch_idx == 0 else False
                    )  # tensors retained to allow backpropagation for torchhviz (for visualizing the graph)
                    optimizer.step()
                    torch.cuda.empty_cache()

                    step3_time = time.time()
                    print(
                        f"(in train_nn) Step 1: {step1_time - start_time:.4f}s, Step 2: {step2_time - step1_time:.4f}s, Step 3: {step3_time - step2_time:.4f}s"
                    )

                    # if epoch == 0 and batch_idx == 0:
                    #     # Note: graphviz is fine for small graphs, but for large graphs it becomes cumbersome so instead opting TensorBoard for the latter
                    #     # graph = torchviz.make_dot(loss, params=dict(model.named_parameters()))
                    #     # file_path = os.path.abspath("data_flow_graph")
                    #     # graph.render(file_path, format="png")
                    #     # print(f"Graph saved as {file_path}.png. You can open it manually.")
                    #
                    #     writer.add_graph(model_to_log, [x_wsi[0], x_omic])
                    #     print(f"Computation graph saved to TensorBoard logs at {log_dir}")

            train_loss = loss_epoch / len(
                train_loader_fold.dataset
            )  # average training loss per sample for the epoch

            scheduler.step()  # step scheduler after each epoch
            print("\n train loss over epoch: ", train_loss)
            end_train_time = time.time()
            train_duration = end_train_time - start_train_time

            # return here for profile
            if opt.profile:
                return model, optimizer
            # check validation for all epochs >= 0
            if epoch >= 0:

                # save model once every 5 epochs
                if (epoch + 1) % 5 == 0 and epoch > 0:
                    checkpoint_path = os.path.join(
                        checkpoint_dir, f"checkpoint_fold_{fold_idx}_epoch_{epoch}.pth"
                    )
                    torch.save(
                        {
                            "epoch": epoch,
                            "fold": fold_idx,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        },
                        checkpoint_path,
                    )
                    print(
                        f"Checkpoint saved at epoch {epoch}, fold {fold_idx}, to {checkpoint_path}"
                    )

                # before validation to get the dynamic weights, but only if the mode is "wsi_omic"

                start_val_time = time.time()

                # get predictions on the validation dataset
                model.eval()
                val_loss_epoch = 0.0
                all_predictions = []
                all_times = []
                all_events = []

                with torch.no_grad():  # inference on the validation data
                    for batch_idx, (
                        tcga_id,
                        days_to_event,
                        event_occurred,
                        x_wsi,
                        x_omic,
                    ) in enumerate(val_loader_fold):
                        # x_wsi is a list of tensors (one tensor for each tile)
                        print(f"Batch size: {len(val_loader_fold.dataset)}")
                        print(
                            f"Validation Batch index: {batch_idx + 1} out of {np.ceil(len(val_loader_fold.dataset) / opt.val_batch_size)}"
                        )

                        x_wsi = [x.to(device) for x in x_wsi]

                        x_omic = x_omic.to(device)

                        days_to_event = days_to_event.to(device)
                        event_occurred = event_occurred.to(device)
                        print("Days to event: ", days_to_event)
                        print("event occurred: ", event_occurred)
                        outputs, wsi_embedding, omic_embedding = model(
                            opt,
                            tcga_id,
                            x_wsi=x_wsi,  # list of tensors (one for each tile)
                            x_omic=x_omic,
                        )
                        loss = joint_loss(
                            outputs.squeeze(),
                            # predictions are not survival outcomes, rather log-risk scores beta*X
                            days_to_event,
                            event_occurred,
                            wsi_embedding,
                            omic_embedding,
                        )  # Cox partial likelihood loss for survival outcome prediction
                        print("\n loss (validation): ", loss.data.item())
                        val_loss_epoch += loss.data.item() * len(tcga_id)
                        all_predictions.append(outputs.squeeze())
                        all_times.append(days_to_event)
                        all_events.append(event_occurred)

                    val_loss = val_loss_epoch / len(val_loader_fold.dataset)

                    all_predictions = torch.cat(all_predictions)
                    all_times = torch.cat(all_times)
                    all_events = torch.cat(all_events)

                    # convert to numpy arrays for CI calculation
                    all_predictions_np = all_predictions.cpu().numpy()
                    all_times_np = all_times.cpu().numpy()
                    all_events_np = all_events.cpu().numpy()
                    event_rate = all_events_np.mean()

                    c_index = concordance_index_censored(
                        all_events_np.astype(bool), all_times_np, all_predictions_np
                    )
                    print(f"Validation loss: {val_loss}, CI: {c_index[0]}")

                    # Calculate cosine similarity every 5 epochs
                    # if (epoch + 1) % 5 == 0 and epoch > 0:
                    # if epoch > -1:
                    #     cosine_metrics, plot_path = analyze_combined_embeddings(
                    #         all_embeddings,
                    #         all_predictions,
                    #         num_bins=2,
                    #         epoch=epoch,
                    #         fold_idx=fold_idx,
                    #         checkpoint_dir=checkpoint_dir,
                    #     )

                    #     wandb.log(
                    #         {f"UMAP Plot": wandb.Image(plot_path)}, step=step_counter
                    #     )
                    #     wandb.log(cosine_metrics, step=step_counter)

                    end_val_time = time.time()
                    val_duration = end_val_time - start_val_time

                    # since fold starts indexing at 0
                    ci_average_by_epoch[epoch] = (
                        (fold_idx) * ci_average_by_epoch[epoch] + c_index[0]
                    ) / (fold_idx + 1)
                    loss_average_by_epoch[epoch] = (
                        fold_idx * loss_average_by_epoch[epoch] + val_loss
                    ) / (fold_idx + 1)

                    mode = (
                        model.module.mode
                        if isinstance(model, nn.DataParallel)
                        else model.mode
                    )
                    if mode == "wsi_omic":
                        wsi_weight_val = (
                            model.module.wsi_weight.item()
                            if isinstance(model.module.wsi_weight, torch.Tensor)
                            else model.module.wsi_weight
                        )
                        omic_weight_val = (
                            model.module.omic_weight.item()
                            if isinstance(model.module.omic_weight, torch.Tensor)
                            else model.module.omic_weight
                        )
                        epoch_metrics = {
                            # Losses
                            "Loss/train_epoch": train_loss,
                            "Loss/val_epoch": val_loss,
                            # Performance metrics
                            "CI/validation": c_index[0],
                            "Event_rate/validation": event_rate,
                            # Performance by epoch
                            "CI/validation/epoch/avg": ci_average_by_epoch[epoch],
                            "Loss/validation/epoch/avg": loss_average_by_epoch[epoch],
                            # Time tracking
                            "Time/train_epoch": train_duration,
                            "Time/val_epoch": val_duration,
                            "Time/total_epoch": train_duration + val_duration,
                            # Learning rate
                            "LR": optimizer.param_groups[0]["lr"],
                            # Metadata
                            "fold": fold_idx,
                            "epoch": epoch,
                            "wsi_weight": wsi_weight_val,
                            "omic_weight": omic_weight_val,
                        }
                    else:
                        epoch_metrics = {
                            # Losses
                            "Loss/train_epoch": train_loss,
                            "Loss/val_epoch": val_loss,
                            # Performance metrics
                            "CI/validation": c_index[0],
                            "Event_rate/validation": event_rate,
                            # Performance by epoch
                            "CI/validation/epoch/avg": ci_average_by_epoch[epoch],
                            "Loss/validation/epoch/avg": loss_average_by_epoch[epoch],
                            # Time tracking
                            "Time/train_epoch": train_duration,
                            "Time/val_epoch": val_duration,
                            "Time/total_epoch": train_duration + val_duration,
                            # Learning rate
                            "LR": optimizer.param_groups[0]["lr"],
                            # Metadata
                            "fold": fold_idx,
                            "epoch": epoch,
                        }
                    # Log all metrics
                    wandb.log(epoch_metrics, step=step_counter)

                    # increment step counter
                    step_counter += 1

        torch.cuda.empty_cache()
        gc.collect()
        print(
            f"Memory cleared after fold {fold_idx} - model will be recreated for next fold"
        )

    wandb.finish()
    return model, optimizer


def print_total_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")
        total_params += param.numel()
    print(f"Total parameters: {total_params / 1e6} million")
    print(
        "Number of trainable params: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )


def denormalize_image(image, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    image = (image * std) + mean
    image = np.clip(image, 0, 1)
    return image


def plot_saliency_maps(
    saliencies,
    x_wsi,
    tcga_id,
    patch_id,
    output_dir,
    threshold=0.8,
):
    # get the first image and saliency map
    saliency = saliencies[0].squeeze().cpu().numpy()
    # image = x_wsi[0].squeeze().permute(1, 2, 0).cpu().numpy()  # convert image to HxWxC and move to cpu
    image = x_wsi[0].squeeze().permute(1, 2, 0).detach().cpu().numpy()
    # denormalize the image based on normalization factors used during transformations of the test images
    mean = [0.70322989, 0.53606487, 0.66096631]
    std = [0.21716536, 0.26081574, 0.20723464]
    image = denormalize_image(image, mean, std)

    # normalize the saliency map to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("original patch")

    # overlay saliency map on the image
    ax = plt.subplot(1, 3, 2)
    img = ax.imshow(image)
    # saliency_overlay = ax.imshow(saliency, cmap="hot", alpha=0.5)
    saliency_overlay = ax.imshow(np.clip(saliency, threshold, 1), cmap="hot", alpha=0.5)
    cbar = plt.colorbar(saliency_overlay, ax=ax)
    cbar.set_label("saliency value", rotation=270, labelpad=15)

    plt.title("saliency map overlay")

    # plot the saliency map alone
    plt.subplot(1, 3, 3)
    plt.imshow(saliency, cmap="hot")
    plt.colorbar(label="saliency value")
    plt.title("saliency map only")

    save_path = os.path.join(
        output_dir, f"saliency_overlay_{tcga_id[0]}_{patch_id}.png"
    )
    plt.savefig(save_path)
    print(f"saved saliency overlay to {save_path}")

    plt.close()


def plot_enhanced_saliency_maps(
    saliencies,
    x_wsi,
    tcga_id,
    patch_id,
    output_dir,
    threshold_percentile=80,
    save_clean_maps=True,
):
    """
    Enhanced saliency map plotting with artifact removal and thresholding

    Args:
        threshold_percentile: Percentile threshold for keeping high-saliency regions (80-90 recommended)
        save_clean_maps: Whether to save cleaned saliency maps
    """
    # Get the first image and saliency map
    saliency = saliencies[0].squeeze().cpu().numpy()
    image = x_wsi[0].squeeze().permute(1, 2, 0).detach().cpu().numpy()

    # Denormalize the image
    mean = [0.70322989, 0.53606487, 0.66096631]
    std = [0.21716536, 0.26081574, 0.20723464]
    image = denormalize_image(image, mean, std)

    # Normalize the saliency map to [0, 1]
    saliency_normalized = (saliency - saliency.min()) / (
        saliency.max() - saliency.min()
    )

    # Calculate threshold based on percentile
    threshold_value = np.percentile(saliency_normalized, threshold_percentile)

    # Create cleaned saliency map (remove artifacts below threshold)
    saliency_clean = saliency_normalized.copy()
    saliency_clean[saliency_clean < threshold_value] = 0

    # Create binary mask for high-importance regions
    high_importance_mask = saliency_normalized >= threshold_value

    plt.figure(figsize=(24, 8))

    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original Patch")
    plt.axis("off")

    # Original saliency overlay
    plt.subplot(1, 4, 2)
    plt.imshow(image)
    plt.imshow(saliency_normalized, cmap="hot", alpha=0.5)
    plt.title("Original Saliency Map")
    plt.axis("off")

    # Cleaned saliency overlay (artifacts removed)
    plt.subplot(1, 4, 3)
    plt.imshow(image)
    plt.imshow(saliency_clean, cmap="hot", alpha=0.6)
    plt.title(f"Cleaned Saliency (>{threshold_percentile}th percentile)")
    plt.axis("off")

    # High-importance regions only
    plt.subplot(1, 4, 4)
    plt.imshow(image)
    # Only show regions above threshold
    masked_saliency = np.ma.masked_where(~high_importance_mask, saliency_clean)
    plt.imshow(masked_saliency, cmap="hot", alpha=0.8)
    plt.title("High-Importance Regions Only")
    plt.axis("off")

    # Save the enhanced plot
    save_path = os.path.join(
        output_dir, f"enhanced_saliency_{tcga_id[0]}_{patch_id}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    if save_clean_maps:
        # Save cleaned saliency map as numpy array for later use
        clean_map_path = os.path.join(
            output_dir, f"clean_saliency_{tcga_id[0]}_{patch_id}.npy"
        )
        np.save(clean_map_path, saliency_clean)

    # Return whether this tile has high-importance regions
    has_important_regions = np.any(high_importance_mask)
    importance_ratio = np.sum(high_importance_mask) / high_importance_mask.size

    return has_important_regions, importance_ratio, threshold_value


def calc_integrated_gradients(
    model, opt, tcga_id, x_omic, x_wsi, baseline=None, steps=10
):
    # baseline = None
    # baseline.shape
    # torch.Size([1, 19962])
    # set_trace()
    if baseline is None:
        print("** using trivial zero baseline for IG **")
        baseline = torch.zeros_like(x_omic).to(x_omic.device)
    else:
        print(
            "** Using mean gene expression values over training samples as the baseline for IG **"
        )
        baseline = torch.from_numpy(baseline).to(x_omic.device)
        baseline = baseline.float()
        baseline = baseline * torch.ones_like(x_omic).to(x_omic.device)
    # set_trace()
    print(f"CALCULATING INTEGRATED GRADIENTS OVER {steps} steps")
    scaled_inputs = [
        baseline + (float(i) / steps) * (x_omic - baseline) for i in range(steps + 1)
    ]
    gradients = []
    steps_index = 0
    for scaled_input in scaled_inputs:
        print("steps_index: ", steps_index)
        scaled_input.requires_grad = True
        with torch.enable_grad():
            output = model(
                opt,
                tcga_id,
                x_wsi=x_wsi,  # list of tensors (one for each tile)
                x_omic=scaled_input,
            )
            print("output: ", output)
            output = output.sum()
            output.backward()
        gradients.append(scaled_input.grad.detach().cpu().numpy())
        steps_index += 1
    # set_trace()
    avg_gradients = np.mean(gradients[:-1], axis=0)
    integrated_grads = (
        x_omic.detach().cpu().numpy() - baseline.detach().cpu().numpy()
    ) * avg_gradients

    return integrated_grads


def test_and_interpret(opt, model, test_loader, device, baseline=None):
    model.eval()
    test_loss_epoch = 0.0
    all_tcga_ids = []
    all_predictions = []
    all_times = []
    all_events = []

    # base directory
    import os

    base_path = Path(opt.output_base_dir)
    # create the directory to save saliency maps if it doesn't exist
    output_dir_saliency = str(base_path / "saliency_maps_6sep")
    os.makedirs(output_dir_saliency, exist_ok=True)

    output_dir_IG = str(base_path / "IG_6sep")
    os.makedirs(output_dir_IG, exist_ok=True)

    # for training, only the last transformer block (block 11) in the WSI encoder was kept trainable
    # see WSIEncoder class in generate_Wsi_embeddings.py

    # Get the CI and the KM plots for the test set
    excluded_ids = [
        "TCGA-05-4395",
        "TCGA-86-8281",
    ]  # contains anomalous time to event and censoring data
    # remove these ids during the input json/h5 file creation
    with torch.no_grad():
        for batch_idx, (
            tcga_id,
            days_to_event,
            event_occurred,
            x_wsi,
            x_omic,
        ) in enumerate(test_loader):
            if tcga_id[0] in excluded_ids:
                print(f"Skipping TCGA ID: {tcga_id}")
                continue

            x_wsi = [x.to(device) for x in x_wsi]
            x_omic = x_omic.to(device)
            days_to_event = days_to_event.to(device)
            event_occurred = event_occurred.to(device)

            # enable gradients only after data loading
            x_wsi = [x.requires_grad_() for x in x_wsi]

            # print(f"Batch size: {len(test_loader.dataset)}")
            # print(
            #     f"Test Batch index: {batch_idx + 1} out of {np.ceil(len(test_loader.dataset) / opt.test_batch_size)}"
            # )
            # # print("TCGA ID: ", tcga_id)
            # print("Days to event: ", days_to_event)
            # print("event occurred: ", event_occurred)

            if opt.calc_saliency_maps is False:
                outputs, wsi_embedding, omic_embedding = model(
                    opt,
                    tcga_id,
                    x_wsi=x_wsi,  # list of tensors (one for each tile)
                    x_omic=x_omic,
                )

            if opt.calc_saliency_maps is True:
                # perform the forward pass without torch.no_grad() to allow gradient computation
                with torch.enable_grad():
                    outputs, wsi_embedding, omic_embedding = model(
                        opt,
                        tcga_id,
                        x_wsi=x_wsi,  # list of tensors (one for each tile)
                        x_omic=x_omic,
                    )

                    # Check and print memory usage after each batch
                    allocated_memory = torch.cuda.memory_allocated(device) / (
                        1024**3
                    )  # in GB
                    reserved_memory = torch.cuda.memory_reserved(device) / (
                        1024**3
                    )  # in GB

                    print(f"After batch {batch_idx + 1}:")
                    print(f"Allocated memory: {allocated_memory:.2f} GB")
                    print(f"Reserved memory: {reserved_memory:.2f} GB")
                    print(
                        f"Free memory: {torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)} bytes"
                    )
                    torch.cuda.empty_cache()

                    # set_trace()
                    # backward pass to compute gradients for saliency maps
                    outputs.backward()  # if outputs is not scalar, reduce it to scalar

                    # saliencies = []  # list of saliency maps corresponding to each image in `x_wsi`
                    patch_idx = 0
                    max_patches = 10
                    print("OBTAINING SALIENCY MAPS")
                    for image in x_wsi:
                        print(
                            f"Generating saliency map for patch index {patch_idx} out of {max_patches}"
                        )
                        if patch_idx >= max_patches:  # limit to 10 patches
                            break
                        if image.grad is not None:
                            saliency, _ = torch.max(image.grad.data.abs(), dim=1)
                            # saliencies.append(saliency)
                            plot_saliency_maps(
                                saliency, image, tcga_id, patch_idx, output_dir_saliency
                            )
                            del saliency
                        else:
                            raise RuntimeError(
                                "Gradients have not been computed for one of the images in x_wsi."
                            )
                        patch_idx += 1
                    # plot_saliency_maps(saliencies, x_wsi, tcga_id)

            if opt.calc_IG is True:
                integrated_grads = calc_integrated_gradients(
                    model, opt, tcga_id, x_omic, x_wsi, baseline=baseline, steps=10
                )
                save_path = os.path.join(
                    output_dir_IG, f"integrated_grads_{tcga_id[0]}.npy"
                )
                np.save(save_path, integrated_grads)
                print(f"Saved integrated gradients for {tcga_id[0]} to {save_path}")
            # set_trace()

            # loss = cox_loss(outputs.squeeze(),
            #                 # predictions are not survival outcomes, rather log-risk scores beta*X
            #                 days_to_event,
            #                 event_occurred)  # Cox partial likelihood loss for survival outcome prediction

            # print("\n loss (test): ", loss.data.item())
            # test_loss_epoch += loss.data.item() * len(tcga_id)
            all_predictions.append(outputs.squeeze().detach().cpu().numpy())
            # all_predictions.append(outputs.squeeze())
            del outputs
            torch.cuda.empty_cache()
            all_tcga_ids.append(tcga_id)
            all_times.append(days_to_event)
            all_events.append(event_occurred)
            model.zero_grad()
            torch.cuda.empty_cache()

        all_predictions_np = [pred.item() for pred in all_predictions]
        all_events_np = torch.stack(all_events).cpu().numpy()
        all_events_bool_np = all_events_np.astype(bool)
        all_times_np = torch.stack(all_times).cpu().numpy()

        c_index = concordance_index_censored(
            all_events_bool_np.ravel(), all_times_np.ravel(), all_predictions_np
        )

        print(f"CI: {c_index[0]}")

    # set_trace()
    # stratify based on the median risk scores
    median_prediction = np.median(all_predictions_np)
    high_risk_idx = all_predictions_np >= median_prediction
    low_risk_idx = all_predictions_np < median_prediction

    # separate the times and events into high and low-risk groups
    high_risk_times = all_times_np[high_risk_idx]
    high_risk_events = all_events_np[high_risk_idx]
    low_risk_times = all_times_np[low_risk_idx]
    low_risk_events = all_events_np[low_risk_idx]

    # initialize the Kaplan-Meier fitter
    kmf_high_risk = KaplanMeierFitter()
    kmf_low_risk = KaplanMeierFitter()

    # fit
    kmf_high_risk.fit(
        high_risk_times, event_observed=high_risk_events, label="High Risk"
    )
    kmf_low_risk.fit(low_risk_times, event_observed=low_risk_events, label="Low Risk")

    # perform the log-rank test
    log_rank_results = logrank_test(
        high_risk_times,
        low_risk_times,
        event_observed_A=high_risk_events,
        event_observed_B=low_risk_events,
    )

    p_value = log_rank_results.p_value
    print(f"Log-Rank Test p-value: {p_value}")
    print(f"Log-Rank Test statistic: {log_rank_results.test_statistic}")

    plt.figure(figsize=(10, 6))
    kmf_high_risk.plot(ci_show=True, color="blue")
    kmf_low_risk.plot(ci_show=True, color="red")
    plt.title(
        "Patient stratification: high risk vs low risk groups based on predicted risk scores\nLog-rank test p-value: {:.4f}".format(
            p_value
        )
    )
    plt.xlabel("Time (days)")
    plt.ylabel("Survival probability")
    plt.legend()
    output_path = str(Path(opt.output_base_dir) / "km_plot_joint_fusion.png")
    plt.savefig(output_path, format="png", dpi=300)

    return None


def evaluate_test_set(model, test_loader, device, opt, excluded_ids=None):
    """
    Shared function for test set evaluation
    """
    if excluded_ids is None:
        excluded_ids = ["TCGA-05-4395", "TCGA-86-8281"]

    # Extract the base model from DataParallel wrapper if it exists
    if isinstance(model, nn.DataParallel):
        test_model = model.module
        print("Removed DataParallel wrapper for testing")
    else:
        test_model = model

    # Move to single GPU and ensure it's in eval mode
    test_model = test_model.to(device)
    test_model.eval()

    test_predictions = []
    test_times = []
    test_events = []

    with torch.no_grad():
        for batch_idx, (
            tcga_id,
            days_to_event,
            event_occurred,
            x_wsi,
            x_omic,
        ) in enumerate(test_loader):
            if tcga_id[0] in excluded_ids:
                print(f"Skipping TCGA ID: {tcga_id}")
                continue

            x_wsi = [x.to(device) for x in x_wsi]
            x_omic = x_omic.to(device)
            days_to_event = days_to_event.to(device)
            event_occurred = event_occurred.to(device)

            print(f"Batch size: {len(test_loader.dataset)}")
            print(
                f"Test Batch index: {batch_idx + 1} out of {np.ceil(len(test_loader.dataset) / opt.test_batch_size)}"
            )
            print("TCGA ID: ", tcga_id)
            print("Days to event: ", days_to_event)
            print("event occurred: ", event_occurred)

            outputs, _, _ = test_model(opt, tcga_id, x_wsi=x_wsi, x_omic=x_omic)

            # Collect results consistently
            test_predictions.append(outputs.squeeze().detach().cpu().numpy())
            test_times.append(days_to_event.cpu().numpy())
            test_events.append(event_occurred.cpu().numpy())

    # Process results consistently
    all_predictions_np = [np.asarray(pred).flatten()[0] for pred in test_predictions]
    all_events_np = np.concatenate(test_events)
    all_times_np = np.concatenate(test_times)
    test_event_rate = all_events_np.mean()

    # Safe CI calculation
    try:
        test_ci = concordance_index_censored(
            all_events_np.astype(bool), all_times_np, all_predictions_np
        )[0]
        print(f"Test CI: {test_ci}")
    except Exception as e:
        print(f"Could not calculate test CI: {e}")
        test_ci = float("nan")

    median_prediction = np.median(all_predictions_np)
    high_risk_idx = all_predictions_np >= median_prediction
    low_risk_idx = all_predictions_np < median_prediction

    # separate the times and events into high and low-risk groups
    high_risk_times = all_times_np[high_risk_idx]
    high_risk_events = all_events_np[high_risk_idx]
    low_risk_times = all_times_np[low_risk_idx]
    low_risk_events = all_events_np[low_risk_idx]

    # initialize the Kaplan-Meier fitter
    kmf_high_risk = KaplanMeierFitter()
    kmf_low_risk = KaplanMeierFitter()

    # fit
    kmf_high_risk.fit(
        high_risk_times, event_observed=high_risk_events, label="High Risk"
    )
    kmf_low_risk.fit(low_risk_times, event_observed=low_risk_events, label="Low Risk")

    # perform the log-rank test
    log_rank_results = logrank_test(
        high_risk_times,
        low_risk_times,
        event_observed_A=high_risk_events,
        event_observed_B=low_risk_events,
    )

    p_value = log_rank_results.p_value
    print(f"Log-Rank Test p-value: {p_value}")
    print(f"Log-Rank Test statistic: {log_rank_results.test_statistic}")
    return test_ci, test_event_rate, log_rank_results.test_statistic, p_value


def train_and_test(opt, h5_file, device):

    from pathlib import Path
    from dotenv import load_dotenv

    env_path: Path = Path.cwd() / ".env"

    load_dotenv(env_path)

    project = os.getenv("WANDB_PROJECT")
    entity = os.getenv("WANDB_ENTITY")

    wandb.init(
        project=project,
        entity=entity,
        config={
            "num_folds": opt.n_folds,
            "num_epochs": opt.num_epochs,
            "batch_size": opt.batch_size,
            "val_batch_size": opt.val_batch_size,
            "test_batch_size": opt.test_batch_size,
            "lr": opt.lr,
            "mlp_layers": opt.mlp_layers,
            "dropout": opt.dropout,
            "fusion_type": opt.fusion_type,
            "joint_embedding": opt.joint_embedding,
            "embedding_dim_wsi": opt.embedding_dim_wsi,
            "embedding_dim_omic": opt.embedding_dim_omic,
            "input_mode": opt.input_mode,
            "stratified_cv": True,
            "use_pretrained_omic": opt.use_pretrained_omic,
            "omic_checkpoint_path": (
                opt.omic_checkpoint_path if opt.use_pretrained_omic else None
            ),
        },
    )

    if opt.scheduler == "cosine_warmer":

        wandb.config.update(
            {
                "scheduler": opt.scheduler,
                "scheduler_params": {
                    "t_0": opt.t_0,
                    "t_mult": opt.t_mult,
                    "eta_min": opt.eta_min,
                },
            },
        )

    elif opt.scheduler == "exponential":

        wandb.config.update(
            {
                "scheduler": opt.scheduler,
                "scheduler_params": {
                    "exp_gamma": opt.exp_gamma,
                },
            },
        )
    elif opt.scheduler == "step_lr":

        wandb.config.update(
            {
                "scheduler": opt.scheduler,
                "scheduler_params": {"exp_gamma": opt.exp_gamma, "step": opt.lr_step},
            }
        )
    current_time = datetime.now()

    # If user provided timestamp then use for consistency
    if opt.timestamp:

        checkpoint_dir = "checkpoints/checkpoint_" + opt.timestamp
        os.makedirs(checkpoint_dir, exist_ok=True)

    else:

        checkpoint_dir = "checkpoints/checkpoint_" + current_time.strftime(
            "%Y-%m-%d-%H-%M-%S"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader, validation_loader, test_loader = create_data_loaders(opt, h5_file)

    validation_dataset = validation_loader.dataset
    test_dataset = test_loader.dataset

    # create a combined loader (validation + test) as the validation data hasn't been used for HPO during training
    test_dataset = ConcatDataset([validation_dataset, test_dataset])
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opt.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    dataset = (
        train_loader.dataset
    )  # use only the training dataset for CV during training [both the validation and test splits can be used for testing]

    total_samples = len(dataset)
    print(f"total training data size: {total_samples} samples")

    # Extract survival data for stratification
    print("Extracting survival data for startified CV...")
    times, events = extract_survival_data(dataset)

    print(f"Survival data summary:")
    print(f"  - Total samples: {len(times)}")
    print(f"  - Events: {events.sum()} ({events.mean():.1%})")
    print(f"  - Censored: {len(events) - events.sum()} ({1 - events.mean():.1%})")
    print(f"  - Median survival time: {np.median(times):.1f} days")
    print(f"  - Time range: {times.min():.1f} - {times.max():.1f} days")

    try:
        folds = create_stratified_survival_folds(
            times=times,
            events=events,
            n_splits=opt.n_folds,
            n_time_bins=10,
        )
        print(f"Successfully created {len(folds)} stratified folds")

    except Exception as e:
        print(f"Error creating stratified folds: {e}")
        # Fallback to regular k-fold if stratified fails
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=opt.n_folds, shuffle=True, random_state=6)
        folds = list(kf.split(np.arange(len(dataset))))
        print("Falling back to regular K-Fold CV")

    num_folds = opt.n_folds
    num_epochs = opt.num_epochs

    # Obtain Validation CI and Loss avg at each epoch
    ci_average_by_epoch = np.zeros(opt.num_epochs, dtype=np.float32)
    loss_average_by_epoch = np.zeros(opt.num_epochs, dtype=np.float32)

    # need to keep fold index at 0 for the sake of the averaging by epoch
    if opt.use_mixed_precision:
        amp_scaler = GradScaler()
        print("Using mixed precision training with StandardScaler")

    # Step counter for wandb
    step_counter = 0
    # Main fold iteration loop
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{len(folds)}")
        print(f"{'='*60}")

        # Print fold statistics
        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")

        # Extract survival data for this fold to verify stratification
        times_train_fold = times[train_idx]
        events_train_fold = events[train_idx]
        times_val_fold = times[val_idx]
        events_val_fold = events[val_idx]

        train_event_rate = events_train_fold.mean()
        val_event_rate = events_val_fold.mean()

        print(f"Train event rate: {train_event_rate:.3f}")
        print(f"Validation event rate: {val_event_rate:.3f}")
        print(f"Train median time: {np.median(times_train_fold):.1f} days")
        print(f"Validation median time: {np.median(times_val_fold):.1f} days")

        print_gpu_memory_usage()  # Monitor memory at start of each fold

        # Create subsets for this fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create data loaders for this fold
        train_loader_fold = DataLoader(
            train_subset,
            batch_size=opt.batch_size,
            collate_fn=mixed_collate,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
        val_loader_fold = DataLoader(
            val_subset,
            collate_fn=mixed_collate,
            batch_size=opt.val_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Compute number of batches per epoch
        num_batches_per_epoch = len(train_loader_fold)
        print(
            f"Number of batches per epoch for fold {fold_idx + 1}: {num_batches_per_epoch}"
        )

        # Update wandb config
        wandb.config.update(
            {f"num_batches_per_epoch_fold_{fold_idx}": num_batches_per_epoch},
            allow_val_change=True,
        )

        model = MultimodalNetwork(
            embedding_dim_wsi=opt.embedding_dim_wsi,
            embedding_dim_omic=opt.embedding_dim_omic,
            mode=opt.input_mode,
            fusion_type=opt.fusion_type,
            joint_embedding_type=opt.joint_embedding,
            mlp_layers=opt.mlp_layers,
            dropout=opt.dropout,
        )

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
            torch.cuda.set_device(0)

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
        if opt.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=opt.exp_gamma
            )
        elif opt.scheduler == "cosine_warmer":
            scheduler = CosineAnnealingWarmRestartsDecay(
                optimizer,
                T_0=opt.t_0,
                T_mult=opt.t_mult,
                eta_min=opt.eta_min,
                decay_factor=opt.decay_factor,
            )

        elif opt.scheduler == "step_lr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=opt.step_lr_step, gamma=opt.step_lr_gamma
            )
        else:
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )

        joint_loss = JointLoss()

        # Initialize and fit the scaler for this fold
        print(f"Fitting scaler for fold {fold_idx + 1}...")
        scaler = StandardScaler()
        for batch_idx, (tcga_id, _, _, _, x_omic) in enumerate(train_loader_fold):
            if batch_idx % 10 == 0:  # Print progress every 10 batches
                print(f"  Fitting batch {batch_idx + 1}/{len(train_loader_fold)}")
            x_omic_np = x_omic.cpu().numpy()
            scaler.partial_fit(x_omic_np)

        # Save the scaler for the current fold
        scaler_path = os.path.join(checkpoint_dir, f"scaler_fold_{fold_idx}.save")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved for fold {fold_idx + 1} at {scaler_path}")

        # EPOCH TRAINING LOOP for this fold
        for epoch in range(opt.num_epochs):
            print(
                f"\n--- Fold {fold_idx + 1}/{len(folds)}, Epoch {epoch + 1}/{opt.num_epochs} ---"
            )
            start_train_time = time.time()
            model.train()

            loss_epoch = 0

            # Add debug info for first batch
            print(f"Starting epoch {epoch}, about to iterate over training batches...")
            print(f"Training loader has {len(train_loader_fold)} batches")
            # log the learning rate at the start of the epoch
            current_lr = optimizer.param_groups[0]["lr"]

            # model training in batches for the train dataloader for the current fold
            for batch_idx, (
                tcga_id,
                days_to_event,
                event_occurred,
                x_wsi,
                x_omic,
            ) in enumerate(train_loader_fold):
                # x_wsi is a list of tensors (one tensor for each tile)
                print(
                    f"Total training samples in fold: {len(train_loader_fold.dataset)}"
                )
                print(f"Batch size: {opt.batch_size}")
                print(f"Batch index: {batch_idx} out of {num_batches_per_epoch}")

                # NOTE: Do not apply standard scaler to omic data
                # x_wsi = [
                #     [tile.to(device) for tile in patient_tiles]
                #     for patient_tiles in x_wsi
                # ]
                x_wsi = [x.to(device) for x in x_wsi]
                x_omic = x_omic.to(device)

                print(f"After scaling - x_omic shape: {x_omic.shape}")
                days_to_event = days_to_event.to(device)
                event_occurred = event_occurred.to(device)

                print("Days to event: ", days_to_event)
                print("event occurred: ", event_occurred)

                optimizer.zero_grad()

                if opt.use_mixed_precision:
                    with autocast():  # should wrap only the forward pass including the loss calculation
                        predictions, wsi_embedding, omic_embedding = model(
                            opt,
                            tcga_id,
                            x_wsi=x_wsi,
                            x_omic=x_omic,  # Now properly scaled
                        )

                        # print(f"Model predictions shape: {predictions.shape}")
                        loss = joint_loss(
                            predictions.squeeze(),
                            days_to_event,
                            event_occurred,
                            wsi_embedding,
                            omic_embedding,
                        )
                        print("\n loss (train, mixed precision): ", loss.data.item())
                        loss_epoch += loss.data.item()

                    # Mixed precision backward pass
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    print(" Not using mixed precision")
                    # model for survival outcome (uses Cox PH partial log likelihood as the loss function)
                    # the model output should be considered as beta*X to be used in the Cox loss function

                    start_time = time.time()

                    predictions, wsi_embedding, omic_embedding = model(
                        opt,
                        tcga_id,
                        x_wsi=x_wsi,  # list of tensors (one for each tile)
                        x_omic=x_omic,
                    )
                    # print(f"predictions: {predictions} from train_test.py")
                    step1_time = time.time()
                    loss = joint_loss(
                        predictions.squeeze(),
                        # predictions are not survival outcomes, rather log-risk scores beta*X
                        days_to_event,
                        event_occurred,
                        wsi_embedding,
                        omic_embedding,
                    )  # Cox partial likelihood loss for survival outcome prediction
                    print("\n loss (train): ", loss.data.item())
                    step2_time = time.time()
                    loss_epoch += (
                        loss.data.item()
                    )  # * len(tcga_id)  # multiplying loss by batch size for accurate epoch averaging
                    # backpropagate loss through the entire model arch upto the inputs
                    loss.backward(
                        retain_graph=True if epoch == 0 and batch_idx == 0 else False
                    )  # tensors retained to allow backpropagation for torchhviz (for visualizing the graph)
                    optimizer.step()
                    torch.cuda.empty_cache()

                    step3_time = time.time()
                    print(
                        f"(in train_nn) Step 1: {step1_time - start_time:.4f}s, Step 2: {step2_time - step1_time:.4f}s, Step 3: {step3_time - step2_time:.4f}s"
                    )

            train_loss = loss_epoch / len(
                train_loader_fold.dataset
            )  # average training loss per sample for the epoch

            scheduler.step()  # step scheduler after each epoch
            print("\n train loss over epoch: ", train_loss)
            end_train_time = time.time()
            train_duration = end_train_time - start_train_time

            # return here for profile
            if opt.profile:
                return model, optimizer
            # check validation for all epochs >= 0
            if epoch >= 0:

                # save model once every 5 epochs
                if (epoch + 1) % 5 == 0 and epoch > 0:
                    checkpoint_path = os.path.join(
                        checkpoint_dir, f"checkpoint_fold_{fold_idx}_epoch_{epoch}.pth"
                    )
                    torch.save(
                        {
                            "epoch": epoch,
                            "fold": fold_idx,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        },
                        checkpoint_path,
                    )
                    print(
                        f"Checkpoint saved at epoch {epoch}, fold {fold_idx}, to {checkpoint_path}"
                    )

                # before validation to get the dynamic weights, but only if the mode is "wsi_omic"

                start_val_time = time.time()

                # get predictions on the validation dataset
                model.eval()
                val_loss_epoch = 0.0
                val_loss = 0.0
                val_ci = 0.0
                val_event_rate = 0.0
                val_duration = 0.0

                val_predictions = []
                val_times = []
                val_events = []
                val_wsi_emb = []
                val_omic_emb = []
                val_tcga_id_list = []

                test_ci = 0.0
                test_predictions = []
                test_times = []
                test_events = []

                with torch.no_grad():  # inference on the validation data
                    for batch_idx, (
                        tcga_id,
                        days_to_event,
                        event_occurred,
                        x_wsi,
                        x_omic,
                    ) in enumerate(val_loader_fold):
                        # x_wsi is a list of tensors (one tensor for each tile)
                        print(f"Batch size: {len(val_loader_fold.dataset)}")
                        print(
                            f"Validation Batch index: {batch_idx + 1} out of {np.ceil(len(val_loader_fold.dataset) / opt.val_batch_size)}"
                        )
                        # x_wsi = [
                        #     [tile.to(device) for tile in patient_tiles]
                        #     for patient_tiles in x_wsi
                        # ]
                        x_wsi = [x.to(device) for x in x_wsi]
                        x_omic = x_omic.to(device)

                        days_to_event = days_to_event.to(device)
                        event_occurred = event_occurred.to(device)
                        print("Days to event: ", days_to_event)
                        print("event occurred: ", event_occurred)
                        outputs, wsi_embedding, omic_embedding = model(
                            opt,
                            tcga_id,
                            x_wsi=x_wsi,  # list of tensors (one for each tile)
                            x_omic=x_omic,
                        )

                        loss = joint_loss(
                            outputs.squeeze(),
                            # predictions are not survival outcomes, rather log-risk scores beta*X
                            days_to_event,
                            event_occurred,
                            wsi_embedding,
                            omic_embedding,
                        )  # Cox partial likelihood loss for survival outcome prediction
                        print("\n loss (validation): ", loss.data.item())
                        val_loss_epoch += loss.data.item() * len(tcga_id)
                        val_predictions.append(outputs.squeeze())
                        val_times.append(days_to_event)
                        val_events.append(event_occurred)

                        val_wsi_emb.append(wsi_embedding)
                        val_omic_emb.append(omic_embedding)
                        val_tcga_id_list.extend(tcga_id)

                    val_loss = val_loss_epoch / len(val_loader_fold.dataset)

                    val_predictions = torch.cat(val_predictions)
                    val_times = torch.cat(val_times)
                    val_events = torch.cat(val_events)

                    wsi_embeddings = torch.cat(val_wsi_emb, dim=0)
                    omic_embeddings = torch.cat(val_omic_emb, dim=0)

                    # convert to numpy arrays for CI calculation
                    val_predictions_np = val_predictions.cpu().numpy()
                    val_times_np = val_times.cpu().numpy()
                    val_events_np = val_events.cpu().numpy()
                    val_event_rate = val_events_np.mean()

                    wsi_np = wsi_embeddings.cpu().numpy()
                    omic_np = omic_embeddings.cpu().numpy()
                    tcga_id_np = np.array(val_tcga_id_list)

                    val_ci = concordance_index_censored(
                        val_events_np.astype(bool), val_times_np, val_predictions_np
                    )[0]
                    print(f"Validation loss: {val_loss}, CI: {val_ci}")

                    end_val_time = time.time()
                    val_duration = end_val_time - start_val_time

                    # save embeddings once every 5 epochs
                    if (epoch + 1) % 5 == 0 and epoch > 0:
                        wsi_checkpoint_path = os.path.join(
                            checkpoint_dir,
                            f"wsi_embedding_fold_{fold_idx}_epoch_{epoch}.npy",
                        )
                        np.save(wsi_checkpoint_path, wsi_np)

                        omic_checkpoint_path = os.path.join(
                            checkpoint_dir,
                            f"omic_embedding_fold_{fold_idx}_epoch_{epoch}.npy",
                        )

                        np.save(omic_checkpoint_path, omic_np)

                        print(
                            f"Embeddings saved at epoch {epoch}, fold {fold_idx}, to {wsi_checkpoint_path} and {omic_checkpoint_path}"
                        )

                        id_checkpoint_path = os.path.join(
                            checkpoint_dir,
                            f"tcga_ids_fold_{fold_idx}_epoch_{epoch}.npy",
                        )

                        np.save(id_checkpoint_path, tcga_id_np)

                test_ci, test_event_rate, test_statistic, p_value = evaluate_test_set(
                    model, test_loader, device, opt
                )

                mode = (
                    model.module.mode
                    if isinstance(model, nn.DataParallel)
                    else model.mode
                )
                actual_model = model.module if hasattr(model, "module") else model
                if mode == "wsi_omic":
                    wsi_weight_val = (
                        actual_model.wsi_weight.item()
                        if isinstance(actual_model.wsi_weight, torch.Tensor)
                        else actual_model.wsi_weight
                    )
                    omic_weight_val = (
                        actual_model.omic_weight.item()
                        if isinstance(actual_model.omic_weight, torch.Tensor)
                        else actual_model.omic_weight
                    )
                    epoch_metrics = {
                        # Losses
                        "Loss/train_epoch": train_loss,
                        "Loss/val_epoch": val_loss,
                        # Performance metrics
                        "CI/validation": val_ci,
                        "CI/test": test_ci,
                        "pvalue/test": p_value,
                        "test_statistic/test": test_statistic,
                        "Event_rate/validation": val_event_rate,
                        "Event_rate/test": test_event_rate,
                        # Time tracking
                        "Time/train_epoch": train_duration,
                        "Time/val_epoch": val_duration,
                        "Time/total_epoch": train_duration + val_duration,
                        # Learning rate
                        "LR": optimizer.param_groups[0]["lr"],
                        # Metadata
                        "fold": fold_idx,
                        "epoch": epoch,
                        "wsi_weight": wsi_weight_val,
                        "omic_weight": omic_weight_val,
                    }
                else:
                    epoch_metrics = {
                        # Losses
                        "Loss/train_epoch": train_loss,
                        "Loss/val_epoch": val_loss,
                        # Performance metrics
                        "CI/validation": val_ci,
                        "CI/test": test_ci,
                        "Event_rate/validation": val_event_rate,
                        "Event_rate/test": test_event_rate,
                        # Time tracking
                        "Time/train_epoch": train_duration,
                        "Time/val_epoch": val_duration,
                        "Time/total_epoch": train_duration + val_duration,
                        # Learning rate
                        "LR": optimizer.param_groups[0]["lr"],
                        # Metadata
                        "fold": fold_idx,
                        "epoch": epoch,
                    }
                # Log all metrics
                wandb.log(epoch_metrics, step=step_counter)

                # increment step counter
                step_counter += 1

        torch.cuda.empty_cache()
        gc.collect()
        print(
            f"Memory cleared after fold {fold_idx} - model will be recreated for next fold"
        )

    wandb.finish()
    return model, optimizer


if __name__ == "__main__":
    # for inference
    import argparse
    import pandas as pd
    from pathlib import Path

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

    # Note: True/False have some issues when used in command line
    opt = parser.parse_args()

    # make sure output_base_dir exists
    os.makedirs(opt.output_base_dir, exist_ok=True)

    device, gpu_ids = setup_model_and_device(opt)
    model = load_model_with_multi_gpu_support(opt, device, gpu_ids)

    model.to(device)
    joint_loss = JointLoss()
    model = load_model_state_dict(model, opt.model_path)

    train_loader, validation_loader, test_loader = create_data_loaders(
        opt, opt.input_mapping_data_path + "mapping_data.h5"
    )
    validation_dataset = validation_loader.dataset
    test_dataset = test_loader.dataset

    # create a combined loader (validation + test) as the validation data hasn't been used for HPO during training
    combined_dataset = ConcatDataset([validation_dataset, test_dataset])
    combined_loader = torch.utils.data.DataLoader(
        dataset=combined_dataset,
        batch_size=opt.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # get the mean expression level for all genes across all the training samples to be used as baseline for x_omic for IG
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

    mean_x_omic = compute_mean_omic_from_h5(
        opt.input_mapping_data_path + "mapping_data.h5"
    )

    test_and_interpret(opt, model, combined_loader, device, baseline=mean_x_omic)
