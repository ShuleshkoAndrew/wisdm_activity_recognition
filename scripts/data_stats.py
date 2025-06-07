"""Script to analyze and visualize preprocessed data statistics."""

import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Activity labels mapping
ACTIVITIES = {
    0: "Walking",
    1: "Jogging",
    2: "Sitting",
    3: "Standing",
    4: "Upstairs",
    5: "Downstairs",
}


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Analyze preprocessed data statistics.

    Args:
        cfg: Hydra configuration
    """
    logger.info("Analyzing preprocessed data statistics")

    # Get project root directory
    project_root = Path(__file__).parents[1]
    processed_data_dir = project_root / "data" / "processed"
    plots_dir = project_root / "plots" / "data_distribution"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data splits
    splits = {}
    for split in ["train", "val", "test"]:
        windows = np.load(processed_data_dir / f"{split}_windows.npy")
        labels = np.load(processed_data_dir / f"{split}_labels.npy")
        splits[split] = (windows, labels)

        # Log basic statistics
        logger.info(f"\n{split.upper()} set statistics:")
        logger.info(f"Number of windows: {len(windows)}")
        logger.info(f"Window shape: {windows.shape}")
        logger.info("Class distribution:")
        for label, count in zip(*np.unique(labels, return_counts=True)):
            logger.info(
                f"  {ACTIVITIES[label]}: {count} ({count/len(labels)*100:.1f}%)"
            )

    # Plot class distribution
    plt.figure(figsize=(12, 6))
    x = np.arange(len(ACTIVITIES))
    width = 0.25

    for i, (split, (_, labels)) in enumerate(splits.items()):
        counts = [np.sum(labels == label) for label in range(len(ACTIVITIES))]
        plt.bar(x + i * width, counts, width, label=split.capitalize())

    plt.xlabel("Activity")
    plt.ylabel("Number of windows")
    plt.title("Class Distribution Across Splits")
    plt.xticks(x + width, ACTIVITIES.values(), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "class_distribution.png")
    plt.close()

    # Plot sample windows for each activity
    train_windows, train_labels = splits["train"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for label in range(len(ACTIVITIES)):
        activity_windows = train_windows[train_labels == label]
        sample_window = activity_windows[0]

        axes[label].plot(sample_window[:, 0], label="x")
        axes[label].plot(sample_window[:, 1], label="y")
        axes[label].plot(sample_window[:, 2], label="z")
        axes[label].set_title(ACTIVITIES[label])
        axes[label].set_xlabel("Time step")
        axes[label].set_ylabel("Acceleration")
        axes[label].legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "sample_windows.png")
    plt.close()

    logger.info(f"\nPlots saved to {plots_dir}")


if __name__ == "__main__":
    main()
