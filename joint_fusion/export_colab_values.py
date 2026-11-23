#!/usr/bin/env python3
"""
export_colab_values.py

Standalone exporter for the c_indices Colab notebook. It does NOT touch bootstrap.py
or any pipeline code -- it only READS the two files your bootstrap run already saved:

    bootstrap_c_indices.npy         -> cindices_jointfusion        (boxplot)
    bootstrap_test_predictions.npz  -> risk_scores/times/events    (Kaplan-Meier)

and prints them as Colab-pasteable `np.array([...])` literals.

Run it:

    python export_colab_values.py                      # newest bootstrap_* folder
    python export_colab_values.py path/to/bootstrap_20260423_174607

Save to a file by redirecting:

    python export_colab_values.py > colab_values.py
"""

import sys
from pathlib import Path

import numpy as np

# Where bootstrap.py writes its bootstrap_* folders.
DEFAULT_RESULTS_DIR = Path(
    "checkpoints/checkpoint_2026-04-07-04-58-17/test_results/best_model_fold_1"
)


def latest_bootstrap_dir(results_dir):
    """Newest bootstrap_* folder that has both saved arrays (name sorts by timestamp)."""
    dirs = [
        d
        for d in sorted(results_dir.glob("bootstrap_*"))
        if (d / "bootstrap_c_indices.npy").exists()
        and (d / "bootstrap_test_predictions.npz").exists()
    ]
    if not dirs:
        sys.exit(
            f"No bootstrap_* folder with bootstrap_c_indices.npy and "
            f"bootstrap_test_predictions.npz under {results_dir}.\n"
            "Run joint_fusion/bootstrap.py first, or pass a folder path as an argument."
        )
    return dirs[-1]


def literal(name, array):
    """`name = np.array([...])` on one line, full precision, comma-separated."""
    return f"{name} = np.array({np.asarray(array).ravel().tolist()!r})"


def main():
    if len(sys.argv) > 1:
        bootstrap_dir = Path(sys.argv[1])
    else:
        bootstrap_dir = latest_bootstrap_dir(DEFAULT_RESULTS_DIR)

    cindices = np.load(bootstrap_dir / "bootstrap_c_indices.npy")
    preds = np.load(bootstrap_dir / "bootstrap_test_predictions.npz", allow_pickle=True)

    print("import numpy as np\n")
    print("# Joint fusion -- bootstrapped C-index (boxplot):")
    print(literal("cindices_jointfusion", cindices))
    print("\n# Joint fusion -- Kaplan-Meier (its own test cohort):")
    print(literal("risk_scores_test_jointfusion", preds["predictions"]))
    print(literal("times_jointfusion", preds["times"].astype(float)))
    print(literal("events_jointfusion", preds["events"].astype(bool)))

    print(f"# source: {bootstrap_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
