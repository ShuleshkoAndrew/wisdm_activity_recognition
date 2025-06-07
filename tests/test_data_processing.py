"""Tests for data processing functionality."""

import pytest
from omegaconf import OmegaConf

from wisdm.data_processing import WISDMDataModule


@pytest.fixture
def config():
    """Create a test configuration."""
    cfg = {"paths": {"data_dir": "data"}}
    return OmegaConf.create(cfg)


def test_data_module(config):
    """Test the data module initialization and basic functionality."""
    # Initialize data module
    data_module = WISDMDataModule(
        data_dir=config.paths.data_dir,
        batch_size=32,
        window_size=100,
        window_overlap=0.5,
    )

    # Setup data
    data_module.setup()

    # Test dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Get a batch and check shapes
    windows, labels = next(iter(train_loader))

    # Basic assertions
    assert len(train_loader) > 0, "Training dataloader is empty"
    assert len(val_loader) > 0, "Validation dataloader is empty"
    assert len(test_loader) > 0, "Test dataloader is empty"

    assert windows.ndim == 3, "Windows should be 3-dimensional (batch, time, features)"
    assert labels.ndim == 1, "Labels should be 1-dimensional"
    assert windows.shape[0] == labels.shape[0], "Batch sizes should match"
    assert windows.shape[1] == 100, "Window size should be 100"
    assert windows.shape[2] == 3, "Should have 3 features (x, y, z acceleration)"
