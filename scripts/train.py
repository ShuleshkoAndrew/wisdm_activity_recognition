"""Training script for the WISDM activity recognition model."""

import logging
import os
from pathlib import Path

import hydra
import mlflow
import pytorch_lightning as pl
import torch
from mlflow.models.signature import infer_signature
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import MLFlowLogger

from wisdm.data_processing.dataset import WISDMDataModule
from wisdm.models.cnn_lstm import CNNLSTM

# Set project root before Hydra config loading
os.environ["PROJECT_ROOT"] = str(Path(__file__).parents[1].absolute())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train model.

    Args:
        cfg: Hydra config
    """
    # Set up MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Create data module
    data_module = WISDMDataModule(
        data_dir=cfg.paths.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        window_size=cfg.model.window_size,
        window_overlap=cfg.model.window_overlap,
        sampling_rate=cfg.model.sampling_rate,
        random_state=cfg.training.seed,
    )

    # Setup data and save scaler
    data_module.setup()
    scaler_path = Path(cfg.paths.models_dir) / "scaler.pkl"
    data_module.preprocessor.save_scaler(scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")

    # Create model
    model = CNNLSTM(
        n_channels=3,
        n_classes=6,
        window_size=cfg.model.window_size,
        hidden_size=cfg.model.hidden_size,
        n_lstm_layers=cfg.model.n_lstm_layers,
        dropout=cfg.model.dropout,
        learning_rate=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
    )

    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.paths.models_dir,
            filename=cfg.training.callbacks.model_checkpoint.filename,
            monitor=cfg.training.callbacks.model_checkpoint.monitor,
            mode=cfg.training.callbacks.model_checkpoint.mode,
            save_top_k=cfg.training.callbacks.model_checkpoint.save_top_k,
        ),
        EarlyStopping(
            monitor=cfg.training.callbacks.early_stopping.monitor,
            patience=cfg.training.callbacks.early_stopping.patience,
            mode=cfg.training.callbacks.early_stopping.mode,
        ),
        RichProgressBar(),
    ]

    # Create logger
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        tags={"model_type": "cnn_lstm"},
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.trainer.max_epochs,
        accelerator=cfg.training.trainer.accelerator,
        devices=cfg.training.trainer.devices,
        precision=cfg.training.trainer.precision,
        deterministic=cfg.training.trainer.deterministic,
        callbacks=callbacks,
        logger=mlf_logger,
        default_root_dir=cfg.paths.models_dir,
    )

    # Train model
    trainer.fit(model, data_module)

    # Test model
    trainer.test(model, data_module)

    # Create input example and signature for MLflow
    input_example = torch.zeros(
        (1, cfg.model.window_size, 3), dtype=torch.float32
    ).numpy()  # Convert to numpy array
    test_dataloader = data_module.test_dataloader()
    batch = next(iter(test_dataloader))
    x, y = batch
    prediction = model(x[:1]).detach().numpy()  # Convert prediction to numpy

    signature = infer_signature(
        x[:1].numpy(),  # Input example
        prediction,  # Output example (already numpy)
    )

    # Log model to MLflow
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name=cfg.mlflow.model_name,
        signature=signature,
        input_example=input_example,
    )


if __name__ == "__main__":
    main()
