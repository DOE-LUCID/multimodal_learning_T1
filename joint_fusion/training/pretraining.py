import torch
import numpy as np
import typing as tp

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold

from joint_fusion.data.datasets import HDF5Dataset


class StratifiedBatchSampler:
    """
    Combines time stratification with risk-set preservation
    """

    def __init__(
        self,
        times,
        events,
        batch_size,
        min_risk_coverage=0.8,
        shuffle=True,
        random_state: int = 40,
    ):
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

        print("Time bin distribution:")
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

    print("Created stratification labels:")
    print(f"  - {len(np.unique(strat_labels))} unique stratification groups")
    print(f"  - Range: {strat_labels.min()} to {strat_labels.max()}")

    return strat_labels


def create_stratified_survival_folds(
    times: np.ndarray,
    events: np.ndarray,
    n_splits: int = 5,
    n_time_bins: int = 5,
    random_state: int = 40,
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


def create_data_loaders(config, h5_file, random_state=40):
    full_dataset = HDF5Dataset(
        h5_file,
        split="train",
        mode=config.model.input_mode,
        train_val_test="train",
    )

    times, events = extract_survival_data(full_dataset)

    batch_sampler = StratifiedBatchSampler(
        times=times,
        events=events,
        batch_size=config.training.batch_size,
        min_risk_coverage=0.8,
        shuffle=True,
        random_state=random_state,
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
            h5_file,
            split="val",
            mode=config.model.input_mode,
            train_val_test="val",
        ),
        batch_size=config.training.val_batch_size,
        shuffle=True,
        num_workers=0,  # 8,
        # prefetch_factor=2,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=HDF5Dataset(
            h5_file,
            split="test",
            mode=config.model.input_mode,
            train_val_test="test",
        ),
        batch_size=config.testing.test_batch_size,
        shuffle=True,
        num_workers=0,  # 1, #4,
        # prefetch_factor=2,
        pin_memory=True,
    )

    return train_loader, validation_loader, test_loader
