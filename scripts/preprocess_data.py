"""Script to preprocess the WISDM dataset."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from wisdm.data_processing.preprocessing import preprocess_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the preprocessing pipeline.

    Args:
        cfg: Hydra configuration object containing preprocessing parameters
    """
    logger.info("Starting data preprocessing")

    # Get project root directory
    project_root = Path(__file__).parents[1]

    # Define paths
    raw_data_dir = project_root / "data" / "raw"
    processed_data_dir = project_root / "data" / "processed"

    # Create processed data directory if it doesn't exist
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Create preprocessing config with shared parameters
    preprocess_cfg = OmegaConf.create(
        {
            "window_size": cfg.window_size,
            "window_overlap": cfg.window_overlap,
            "sampling_rate": cfg.sampling_rate,
            "random_seed": cfg.random_seed,
            "val_split": cfg.preprocessing.val_split,
            "test_split": cfg.preprocessing.test_split,
        }
    )

    # Run preprocessing
    try:
        preprocess_dataset(
            raw_data_dir=raw_data_dir,
            processed_data_dir=processed_data_dir,
            config=preprocess_cfg,
        )
        logger.info("Data preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()
