# WISDM Activity Recognition

This project implements a deep learning model for human activity recognition using the WISDM v1.1 dataset. The model uses a hybrid CNN-LSTM architecture to classify activities from accelerometer data.

## Dataset

The WISDM v1.1 dataset contains accelerometer data collected from smartwatches and smartphones, with the following characteristics:
- 6 activities: Walking, Jogging, Sitting, Standing, Upstairs, Downstairs
- 3 axes of acceleration (x, y, z)
- Data collected at 20Hz sampling rate
- Multiple users for better generalization

## Setup

1. Environment Setup:
```bash
# Install Poetry if not installed
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to your PATH (required after installation)
export PATH="/Users/$USER/.local/bin:$PATH"

# Verify Poetry installation
poetry --version

# Set Python version and install dependencies
poetry env use python3.10
poetry install

# Activate the virtual environment
source $(poetry env info --path)/bin/activate

# To deactivate the environment when done
deactivate
```

2. Setup Pre-commit Hooks:
```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run pre-commit checks on all files
poetry run pre-commit run -a
```

3. Set Environment Variables:
```bash
export PROJECT_ROOT=$(pwd)
export MLFLOW_TRACKING_URI=file://${PROJECT_ROOT}/mlruns
```

## Data Loading and Preprocessing

1. Download Dataset:
```bash
# Download and extract WISDM dataset
poetry run python scripts/download_data.py
```

This script will:
- Create necessary data directories
- Download the latest WISDM dataset
- Extract it to data/raw/
- Clean up the archive file

2. Data Preprocessing Steps:
```bash
# Run preprocessing script
poetry run python scripts/preprocess_data.py
```

This will:
- Load raw accelerometer data from WISDM dataset
- Clean and validate the data
- Create sliding windows (5 seconds with 50% overlap)
- Apply standard scaling
- Split data by user ID (train/val/test)
- Save processed datasets to `data/processed/`

Key preprocessing parameters (configurable in `conf/data_loading/default.yaml`):
```yaml
window_size: 100  # 5 seconds at 20Hz
window_overlap: 0.5  # 50% overlap
train_split: 0.7
val_split: 0.15
test_split: 0.15
random_seed: 42
```

3. Verify Processed Data:
```bash
# Check processed data statistics
poetry run python scripts/data_stats.py
```

This generates:
- Data distribution plots in `plots/data_distribution/`
- Class balance statistics
- Train/val/test split summary
- Data quality metrics

## Model Architecture

The model uses a hybrid CNN-LSTM architecture:
- CNN layers for feature extraction from raw accelerometer data
- Bidirectional LSTM layers for temporal dependencies
- Dropout for regularization
- Final fully connected layer for classification

Key parameters:
- Input: (batch_size, window_size=100, channels=3)
- CNN layers: [64, 128] with kernel size 3
- LSTM: 2 layers, hidden size 64, bidirectional
- Dropout: 0.5

## Training

1. Data Preprocessing:
   - Sliding windows of 5 seconds (100 samples at 20Hz)
   - 50% window overlap
   - Standard scaling of accelerometer values
   - Stratified split by user ID (70% train, 15% val, 15% test)

2. Training Configuration (`conf/training/default.yaml`):
   - Batch size: 32
   - Max epochs: 100
   - Early stopping patience: 10
   - Learning rate: 0.001 with ReduceLROnPlateau scheduler
   - Weight decay: 0.0001
   - Loss function: Cross-entropy
   - Metrics: Accuracy and F1-score

3. Run Training:
```bash
poetry run python scripts/train.py
```

Training outputs:
- Model checkpoints saved to `models/best_model.ckpt`
- Scaler saved to `models/scaler.pkl`
- MLflow logs in `mlruns/`
- Training metrics and plots in `plots/`

## MLflow Tracking

The training process is tracked using MLflow, logging:
- Metrics: train/val/test loss, accuracy, F1-score
- Parameters: model architecture, training config
- Artifacts: model checkpoints, plots
- Run information: git commit, timestamps

### MLflow Server Configuration

You can use either a local file system or a remote MLflow tracking server:

1. Local File System (default):
```bash
# Set environment variable to use local mlruns directory
export PROJECT_ROOT=$(pwd)
export MLFLOW_TRACKING_URI="file://${PROJECT_ROOT}/mlruns"
```

2. Remote MLflow Server:
```bash
# Set environment variable to use remote server
export MLFLOW_TRACKING_URI="http://127.0.0.1:8080"

# Start MLflow server (in a separate terminal)
poetry run mlflow server --host 127.0.0.1 --port 8080
```

View the MLflow UI:
- For local tracking: `mlflow ui --port 5000`
- For remote server: Open http://127.0.0.1:8080 in your browser

### Plotting Training Metrics

To visualize training metrics from MLflow runs, use the plotting script:
```bash
# Plot metrics from the latest run that has metrics
poetry run python scripts/plot_metrics_simple.py

# Plot metrics from a specific run
poetry run python scripts/plot_metrics_simple.py plotting.run_id=YOUR_RUN_ID

# Change plot style
poetry run python scripts/plot_metrics_simple.py plotting.style=whitegrid

# Save plots to custom directory
poetry run python scripts/plot_metrics_simple.py plotting.output_dir=custom/path
```

The script will:
- Automatically find a run with metrics if none specified
- Generate separate plots for:
  - Loss curves (train/val)
  - Accuracy curves (train/val/test)
  - F1-score curves (train/val/test)
- Save plots to the specified output directory (default: plots/)

## Model Export

Export trained PyTorch model to ONNX:
```bash
poetry run python scripts/export_model.py
```

This creates:
- ONNX model at `models/onnx/model.onnx`
- Input/output specs in model metadata

## Inference

1. Configure Inference (`conf/inference/default.yaml`):
   - Input data path
   - Output predictions path
   - Batch size
   - Model path (ONNX or checkpoint)

2. Run Inference:
```bash
poetry run python scripts/infer.py
```

Outputs:
- Predictions saved to `data/predictions.csv`
- Format: window_id, timestamp, predicted_activity

## Project Structure

```
wisdm_project/
├── conf/                   # Hydra configuration files
│   ├── data_loading/      # Data preprocessing config
│   ├── model/             # Model architecture config
│   ├── training/          # Training parameters
│   └── inference/         # Inference settings
├── data/                  # Dataset and predictions
│   ├── raw/              # Original WISDM dataset
│   └── processed/        # Preprocessed data
├── models/                # Saved models
│   ├── checkpoints/      # PyTorch checkpoints
│   └── onnx/            # Exported ONNX models
├── wisdm/                # Source code
│   ├── data_processing/ # Data preprocessing
│   └── models/         # Model architectures
├── scripts/             # Training and inference scripts
└── plots/              # Training metrics plots
```

## Dependencies

Main dependencies:
- PyTorch & PyTorch Lightning
- MLflow for experiment tracking
- Hydra for configuration
- NumPy & Pandas for data processing
- ONNX for model export

## License

This project is licensed under the MIT License.
