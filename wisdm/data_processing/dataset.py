"""PyTorch Dataset and Lightning DataModule for WISDM data."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import WISDMPreprocessor

logger = logging.getLogger(__name__)


class WISDMDataset(Dataset):
    """PyTorch Dataset for WISDM data."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        """Initialize dataset.

        Args:
            windows: Array of shape (n_windows, window_size, 3)
            labels: Array of shape (n_windows,)
        """
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index.

        Args:
            idx: Index

        Returns:
            Tuple of (window, label)
        """
        return self.windows[idx], self.labels[idx]


class WISDMDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for WISDM data."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        window_size: int = 100,
        window_overlap: float = 0.5,
        sampling_rate: int = 20,
        random_state: int = 42,
    ):
        """Initialize data module.

        Args:
            data_dir: Path to data directory
            batch_size: Batch size
            num_workers: Number of workers for data loading
            window_size: Number of samples in each window
            window_overlap: Overlap between windows (0.0 to 1.0)
            sampling_rate: Sampling rate in Hz
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.preprocessor = WISDMPreprocessor(
            window_size=window_size,
            window_overlap=window_overlap,
            sampling_rate=sampling_rate,
            random_state=random_state,
        )

        self.data_train: Optional[WISDMDataset] = None
        self.data_val: Optional[WISDMDataset] = None
        self.data_test: Optional[WISDMDataset] = None

    def prepare_data(self) -> None:
        """Download data if needed. This method is called only from a single GPU."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Set up data splits.

        Args:
            stage: Either 'fit' or 'test'
        """
        if stage == "fit" or stage is None:
            # Load and preprocess data
            raw_data_path = (
                self.data_dir / "raw" / "WISDM_ar_v1.1" / "WISDM_ar_v1.1_raw.txt"
            )
            df = self.preprocessor.load_data(raw_data_path)
            windows, labels, users = self.preprocessor.create_windows(df)
            windows = self.preprocessor.normalize_windows(windows)
            splits = self.preprocessor.split_data(windows, labels, users)

            # Create datasets
            self.data_train = WISDMDataset(*splits["train"])
            self.data_val = WISDMDataset(*splits["val"])
            self.data_test = WISDMDataset(*splits["test"])

    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
