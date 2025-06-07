"""Inference script for WISDM activity recognition."""

import logging
import os
from pathlib import Path
from typing import List

import hydra
import numpy as np
import onnxruntime as ort
import pandas as pd
from omegaconf import DictConfig

from wisdm.data_processing import WISDMPreprocessor

# Set project root before Hydra config loading
os.environ["PROJECT_ROOT"] = str(Path(__file__).parents[1].absolute())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_onnx_model(model_path: str) -> ort.InferenceSession:
    """Load ONNX model.

    Args:
        model_path: Path to ONNX model

    Returns:
        ONNX inference session
    """
    # ONNX Runtime options
    options = ort.SessionOptions()
    options.enable_mem_pattern = True
    options.enable_cpu_mem_arena = True

    # Create inference session
    session = ort.InferenceSession(
        model_path,
        options,
        providers=["CPUExecutionProvider"],
    )

    return session


def predict_activities(
    session: ort.InferenceSession,
    windows: np.ndarray,
) -> List[str]:
    """Predict activities for windows.

    Args:
        session: ONNX inference session
        windows: Array of shape (n_windows, window_size, 3)

    Returns:
        List of predicted activity labels
    """
    # Get input name
    input_name = session.get_inputs()[0].name

    # Run inference
    outputs = session.run(None, {input_name: windows.astype(np.float32)})
    predictions = outputs[0]

    # Convert to activity labels
    activity_map = {v: k for k, v in WISDMPreprocessor.ACTIVITIES.items()}
    labels = [activity_map[pred] for pred in predictions.argmax(axis=1)]

    return labels


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run inference on new data.

    Args:
        cfg: Hydra config
    """
    # Load ONNX model
    model_path = Path(cfg.paths.onnx_dir) / "model.onnx"
    session = load_onnx_model(str(model_path))

    # Create preprocessor and load scaler
    preprocessor = WISDMPreprocessor(
        window_size=cfg.model.window_size,
        window_overlap=cfg.model.window_overlap,
        sampling_rate=cfg.model.sampling_rate,
    )
    scaler_path = Path(cfg.paths.models_dir) / "scaler.pkl"
    preprocessor.load_scaler(scaler_path)
    logger.info(f"Loaded scaler from {scaler_path}")

    # Load and preprocess data
    data_path = Path(cfg.inference.data_path)
    df = preprocessor.load_data(data_path)
    windows, _, _ = preprocessor.create_windows(df)
    windows = preprocessor.normalize_windows(windows, fit=False)

    # Run inference
    predictions = predict_activities(session, windows)

    # Create output DataFrame
    results = pd.DataFrame(
        {
            "window_id": range(len(predictions)),
            "predicted_activity": predictions,
        }
    )

    # Save results
    output_path = Path(cfg.inference.output_path)
    results.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
