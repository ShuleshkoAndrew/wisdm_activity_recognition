"""Data preprocessing functionality for WISDM dataset."""

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class WISDMPreprocessor:
    """Preprocessor for WISDM dataset."""

    ACTIVITIES = {
        "Walking": 0,
        "Jogging": 1,
        "Sitting": 2,
        "Standing": 3,
        "Upstairs": 4,
        "Downstairs": 5,
    }

    def __init__(
        self,
        window_size: int = 100,
        window_overlap: float = 0.5,
        sampling_rate: int = 20,
        random_state: int = 42,
    ):
        """Initialize preprocessor.

        Args:
            window_size: Number of samples in each window
            window_overlap: Overlap between windows (0.0 to 1.0)
            sampling_rate: Sampling rate in Hz
            random_state: Random seed for reproducibility
        """
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.sampling_rate = sampling_rate
        self.random_state = random_state
        self.scaler = StandardScaler()

    def save_scaler(self, path: str) -> None:
        """Save scaler to file.

        Args:
            path: Path to save scaler
        """
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    def load_scaler(self, path: str) -> None:
        """Load scaler from file.

        Args:
            path: Path to load scaler from
        """
        with open(path, "rb") as f:
            self.scaler = pickle.load(f)

    def load_data(self, data_path: Path) -> pd.DataFrame:
        """Load WISDM dataset.

        Args:
            data_path: Path to raw data file

        Returns:
            DataFrame with columns [user, activity, timestamp, x, y, z]
        """
        # Read the file and remove semicolons
        with open(data_path, "r") as f:
            lines = [line.strip().rstrip(";") for line in f if line.strip()]

        # Parse each line into a list of values
        data = []
        for line in lines:
            try:
                values = line.split(",")
                if len(values) == 6:  # Only take lines with correct number of fields
                    data.append(values)
            except Exception:
                logger.warning(f"Failed to read {data_path}, retrying...")
                continue

        # Create DataFrame
        df = pd.DataFrame(
            data, columns=["user", "activity", "timestamp", "x", "y", "z"]
        )

        # Convert numeric columns
        df["user"] = pd.to_numeric(df["user"])
        df["timestamp"] = pd.to_numeric(df["timestamp"])
        df["x"] = pd.to_numeric(df["x"])
        df["y"] = pd.to_numeric(df["y"])
        df["z"] = pd.to_numeric(df["z"])

        return df

    def create_windows(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sliding windows from accelerometer data.

        Args:
            df: DataFrame with columns [user, activity, timestamp, x, y, z]

        Returns:
            Tuple of (windows, labels, user_ids)
            - windows: Array of shape (n_windows, window_size, 3)
            - labels: Array of shape (n_windows,)
            - user_ids: Array of shape (n_windows,)
        """
        stride = int(self.window_size * (1 - self.window_overlap))
        windows, labels, users = [], [], []

        for user_id in df["user"].unique():
            user_data = df[df["user"] == user_id]

            for activity in user_data["activity"].unique():
                activity_data = user_data[user_data["activity"] == activity]
                xyz_data = activity_data[["x", "y", "z"]].values

                for start in range(0, len(xyz_data) - self.window_size, stride):
                    window = xyz_data[start : start + self.window_size]
                    windows.append(window)
                    labels.append(self.ACTIVITIES[activity])
                    users.append(user_id)

        return (
            np.array(windows),
            np.array(labels),
            np.array(users),
        )

    def normalize_windows(self, windows: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize windows using StandardScaler.

        Args:
            windows: Array of shape (n_windows, window_size, 3)
            fit: Whether to fit the scaler or just transform

        Returns:
            Normalized windows array
        """
        n_windows, window_size, n_features = windows.shape
        flat_windows = windows.reshape(-1, n_features)

        if fit:
            flat_normalized = self.scaler.fit_transform(flat_windows)
        else:
            flat_normalized = self.scaler.transform(flat_windows)

        return flat_normalized.reshape(n_windows, window_size, n_features)

    def split_data(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        users: np.ndarray,
        val_size: float = 0.15,
        test_size: float = 0.15,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Split data into train/val/test sets stratified by user and activity.

        Args:
            windows: Array of shape (n_windows, window_size, 3)
            labels: Array of shape (n_windows,)
            users: Array of shape (n_windows,)
            val_size: Validation set size (0.0 to 1.0)
            test_size: Test set size (0.0 to 1.0)

        Returns:
            Dict with train/val/test splits
        """
        # First split out test set
        train_idx, test_idx = train_test_split(
            np.arange(len(users)),
            test_size=test_size,
            stratify=users,
            random_state=self.random_state,
        )

        # Then split training data into train/val
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_size / (1 - test_size),
            stratify=users[train_idx],
            random_state=self.random_state,
        )

        splits = {
            "train": (windows[train_idx], labels[train_idx]),
            "val": (windows[val_idx], labels[val_idx]),
            "test": (windows[test_idx], labels[test_idx]),
        }

        return splits


def preprocess_dataset(
    raw_data_dir: Path,
    processed_data_dir: Path,
    config: Dict,
) -> None:
    """Preprocess WISDM dataset.

    Args:
        raw_data_dir: Directory containing raw data files
        processed_data_dir: Directory to save processed data
        config: Configuration dictionary with preprocessing parameters
    """
    # Initialize preprocessor
    preprocessor = WISDMPreprocessor(
        window_size=config.window_size,
        window_overlap=config.window_overlap,
        sampling_rate=config.sampling_rate,
        random_state=config.random_seed,
    )

    # Load and preprocess data
    raw_data_file = raw_data_dir / "WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
    df = preprocessor.load_data(raw_data_file)

    # Create windows
    windows, labels, users = preprocessor.create_windows(df)

    # Normalize windows
    normalized_windows = preprocessor.normalize_windows(windows, fit=True)

    # Split data
    splits = preprocessor.split_data(
        normalized_windows,
        labels,
        users,
        val_size=config.val_split,
        test_size=config.test_split,
    )

    # Save processed data
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    for split_name, (split_windows, split_labels) in splits.items():
        np.save(processed_data_dir / f"{split_name}_windows.npy", split_windows)
        np.save(processed_data_dir / f"{split_name}_labels.npy", split_labels)

    # Save scaler
    preprocessor.save_scaler(str(processed_data_dir / "scaler.pkl"))
