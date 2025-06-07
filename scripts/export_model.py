"""Script to export trained PyTorch model to ONNX format."""

import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from wisdm.models.cnn_lstm import CNNLSTM

# Set project root before Hydra config loading
os.environ["PROJECT_ROOT"] = str(Path(__file__).parents[1].absolute())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Export model to ONNX format.

    Args:
        cfg: Hydra config
    """
    # Load trained model
    checkpoint_path = Path(cfg.paths.models_dir) / "best_model.ckpt"
    model = CNNLSTM.load_from_checkpoint(checkpoint_path)
    model.eval()
    model = model.cpu()  # Move model to CPU

    # Create dummy input
    dummy_input = torch.randn(
        1,  # batch_size
        cfg.model.window_size,  # sequence length
        3,  # number of channels (x, y, z)
    )

    # Create ONNX directory if it doesn't exist
    onnx_dir = Path(cfg.paths.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    onnx_path = onnx_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info(f"Model exported to {onnx_path}")


if __name__ == "__main__":
    main()
