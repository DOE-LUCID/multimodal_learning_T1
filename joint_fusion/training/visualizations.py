from .pretraining import _create_stratification_labels

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


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
