import torch
import numpy as np
import typing as tp

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold

from joint_fusion.utils.utils import mixed_collate
from joint_fusion.data.datasets import HDF5Dataset


class StratifiedBatchSampler:
    """
    Stratified batch sampler for survival data with partial Cox loss.

    Design goals:
      1. Each batch mirrors the training marginal distribution over
         (time_bin x event_status) strata — this keeps gradient estimates
         low-variance and unbiased, which is the correct justification for
         stratified sampling under SGD.
      2. Each batch is guaranteed a minimum number of events so the partial
         Cox likelihood has enough risk-set signal to be meaningful.
      3. No explicit risk-set membership forcing.

    Args:
        times:             Array of survival/censoring times.
        events:            Binary event indicator array.
        batch_size:        Number of samples per batch.
        n_time_bins:       Number of time strata (combined with event status
                           gives 2 * n_time_bins strata total).
        min_events_per_batch: Hard floor on events per batch. Defaults to
                           max(1, round(batch_size * observed_event_rate)).
                           Raise this if batches are too small for stable
                           partial Cox gradients.
        shuffle:           Re-permute within strata each epoch.
        random_state:      Seed for reproducibility.
    """

    def __init__(
        self,
        times,
        events,
        batch_size: int,
        n_time_bins: int = 5,
        min_events_per_batch: tp.Optional[int] = None,
        shuffle: bool = True,
        random_state: int = 40,
    ):
        self.times = np.array(times, dtype=float)
        self.events = np.array(events, dtype=int)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        n = len(self.times)
        observed_event_rate = self.events.mean()

        # Default min events: match training event rate, at least 1
        if min_events_per_batch is None:
            self.min_events_per_batch = max(1, round(batch_size * observed_event_rate))
        else:
            self.min_events_per_batch = min_events_per_batch

        # Build strata: time_bin x event_status -> stratum label
        # Adjust n_time_bins if we have fewer unique times than requested
        n_unique_times = len(np.unique(self.times))
        if n_unique_times < n_time_bins:
            print(
                f"[StratifiedBatchSampler] Only {n_unique_times} unique times; "
                f"reducing n_time_bins from {n_time_bins} to {n_unique_times}."
            )
            n_time_bins = n_unique_times

        self.n_time_bins = n_time_bins

        try:
            disc = KBinsDiscretizer(
                n_bins=n_time_bins,
                encode="ordinal",
                strategy="quantile",
                subsample=None,
            )
            time_bins = (
                disc.fit_transform(self.times.reshape(-1, 1)).flatten().astype(int)
            )
        except Exception as e:
            print(
                f"[StratifiedBatchSampler] KBinsDiscretizer failed ({e}); "
                "falling back to manual quantile bins."
            )
            edges = np.unique(
                np.quantile(self.times, np.linspace(0, 1, n_time_bins + 1))
            )
            time_bins = np.clip(np.digitize(self.times, edges[1:]), 0, n_time_bins - 1)

        # Stratum label: encodes both time position and event status
        # Range: [0, 2 * n_time_bins)
        self.strat_labels = time_bins * 2 + self.events

        # Build per-stratum index lists (these are used for proportional sampling)
        unique_strata, strata_counts = np.unique(self.strat_labels, return_counts=True)
        self.strata = {s: np.where(self.strat_labels == s)[0] for s in unique_strata}
        # Proportion of each stratum in the full dataset
        self.strata_proportions = {
            s: cnt / n for s, cnt in zip(unique_strata, strata_counts)
        }

        # Separate event / censored index pools for the minimum-events guarantee
        self.event_indices = np.where(self.events == 1)[0]
        self.censored_indices = np.where(self.events == 0)[0]

        print(
            f"[StratifiedBatchSampler] {n} samples | "
            f"event rate={observed_event_rate:.3f} | "
            f"{len(unique_strata)} strata | "
            f"min_events_per_batch={self.min_events_per_batch}"
        )

    def __iter__(self):
        # Fresh per-epoch permutation within each stratum
        if self.shuffle:
            strata_pools = {
                s: self.rng.permutation(idx) for s, idx in self.strata.items()
            }
        else:
            strata_pools = {s: idx.copy() for s, idx in self.strata.items()}

        # Pointers into each stratum pool
        strata_ptrs = {s: 0 for s in strata_pools}

        total = len(self.times)
        n_batches = total // self.batch_size  # drop last incomplete batch

        for _ in range(n_batches):
            batch = self._create_stratified_batch(strata_pools, strata_ptrs)
            if batch is not None:
                yield batch

    def __len__(self):
        return len(self.times) // self.batch_size

    def _create_stratified_batch(self, strata_pools, strata_ptrs):
        """
        Build one batch by:
          1. Proportional sampling from each stratum (mirrors training marginal).
          2. Topping up from the event pool if the minimum-events floor isn't met.
          3. Filling any remaining slots uniformly from all strata.
        """
        batch_indices = []
        n_events_so_far = 0

        # Step 1: proportional draw from each stratum
        slots_per_stratum = self._compute_stratum_slots()

        for stratum, n_slots in slots_per_stratum.items():
            pool = strata_pools[stratum]
            ptr = strata_ptrs[stratum]

            # Wrap pointer if stratum is exhausted mid-epoch
            available = len(pool) - ptr
            if available <= 0:
                strata_ptrs[stratum] = 0
                ptr = 0
                available = len(pool)

            take = min(n_slots, available)
            selected = pool[ptr : ptr + take]
            strata_ptrs[stratum] += take

            batch_indices.extend(selected.tolist())
            n_events_so_far += self.events[selected].sum()

        # Step 2: enforce minimum events per batch
        if n_events_so_far < self.min_events_per_batch:
            shortfall = self.min_events_per_batch - n_events_so_far
            current_set = set(batch_indices)

            # Draw from event pool, excluding already-selected indices
            candidate_events = [i for i in self.event_indices if i not in current_set]
            if self.shuffle:
                candidate_events = self.rng.permutation(candidate_events).tolist()

            top_up = candidate_events[:shortfall]
            batch_indices.extend(top_up)
            n_events_so_far += len(top_up)

        # Step 3: fill remaining slots up to batch_size
        remaining = self.batch_size - len(batch_indices)
        if remaining > 0:
            current_set = set(batch_indices)
            all_indices = np.arange(len(self.times))
            leftover = [i for i in all_indices if i not in current_set]
            if self.shuffle:
                leftover = self.rng.permutation(leftover).tolist()
            batch_indices.extend(leftover[:remaining])

        if len(batch_indices) < self.batch_size:
            # Dataset nearly exhausted — skip this batch
            return None

        # Final shuffle within batch so model doesn't see time-sorted input
        batch_indices = batch_indices[: self.batch_size]
        if self.shuffle:
            batch_indices = self.rng.permutation(batch_indices).tolist()

        return batch_indices

    def _compute_stratum_slots(self) -> tp.Dict[int, int]:
        """
        Allocate batch slots proportionally to each stratum's share of the data.
        Uses floor allocation then distributes remainders to largest-remainder strata
        so the total always equals batch_size exactly.
        """
        raw = {
            s: self.strata_proportions[s] * self.batch_size
            for s in self.strata_proportions
        }
        floors = {s: int(v) for s, v in raw.items()}
        remainders = {s: raw[s] - floors[s] for s in raw}

        allocated = sum(floors.values())
        shortfall = self.batch_size - allocated

        # Distribute remaining slots to strata with largest fractional remainders
        sorted_by_remainder = sorted(remainders, key=remainders.get, reverse=True)
        for s in sorted_by_remainder[:shortfall]:
            floors[s] += 1

        return floors


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
        min_events_per_batch=1,
        shuffle=True,
        random_state=random_state,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=full_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,  # 8,
        collate_fn=mixed_collate,
        # prefetch_factor=2,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=HDF5Dataset(
            h5_file,
            split="test",
            mode=config.model.input_mode,
            train_val_test="test",
            return_tile_names=True,
        ),
        batch_size=config.testing.test_batch_size,
        shuffle=False,  # must be false otherwise attention map recontruction issues.
        collate_fn=mixed_collate,
        num_workers=0,  # 1, #4,
        # prefetch_factor=2,
        pin_memory=True,
    )

    return train_loader, test_loader
