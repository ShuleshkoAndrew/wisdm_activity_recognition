"""CNN-LSTM model for human activity recognition."""

from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score


class CNNLSTM(pl.LightningModule):
    """CNN-LSTM model for human activity recognition."""

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 6,
        window_size: int = 100,
        hidden_size: int = 64,
        n_lstm_layers: int = 2,
        dropout: float = 0.5,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
    ):
        """Initialize model.

        Args:
            n_channels: Number of input channels (accelerometer axes)
            n_classes: Number of activity classes
            window_size: Number of samples in each window
            hidden_size: Size of LSTM hidden state
            n_lstm_layers: Number of LSTM layers
            dropout: Dropout probability
            learning_rate: Learning rate
            weight_decay: Weight decay factor
        """
        super().__init__()
        self.save_hyperparameters()

        # CNN layers
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=64,  # From last CNN layer
            hidden_size=hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(dropout)

        # Final classification layer
        self.fc = nn.Linear(hidden_size * 2, n_classes)  # *2 for bidirectional

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=n_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=n_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_features)

        Returns:
            Model output of shape (batch_size, n_classes)
        """
        # CNN layers
        x = x.transpose(1, 2)  # (batch_size, n_features, sequence_length)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.transpose(1, 2)  # (batch_size, sequence_length, n_features)

        # LSTM layers
        x, _ = self.lstm(x)
        x = self.dropout2(x)
        x = x[:, -1, :]  # Take only the last output

        # Dense layers
        x = self.fc(x)
        return x

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Training step.

        Args:
            batch: Tuple of (windows, labels)
            batch_idx: Batch index

        Returns:
            Dict with loss and metrics
        """
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Log metrics
        self.train_acc(y_hat, y)
        self.train_f1(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        self.log("train_f1", self.train_f1, prog_bar=True)

        return {"loss": loss}

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step.

        Args:
            batch: Tuple of (windows, labels)
            batch_idx: Batch index

        Returns:
            Dict with loss and metrics
        """
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Log metrics
        self.val_acc(y_hat, y)
        self.val_f1(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)

        return {"val_loss": loss}

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step.

        Args:
            batch: Tuple of (windows, labels)
            batch_idx: Batch index

        Returns:
            Dict with metrics
        """
        x, y = batch
        y_hat = self(x)

        # Log metrics
        self.test_acc(y_hat, y)
        self.test_f1(y_hat, y)
        self.log("test_acc", self.test_acc)
        self.log("test_f1", self.test_f1)

        return {"y_pred": y_hat.argmax(dim=1), "y_true": y}

    def configure_optimizers(self) -> Dict:
        """Configure optimizers and learning rate schedulers.

        Returns:
            Dict with optimizer and scheduler config
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
