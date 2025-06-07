"""Script to plot training metrics from MLflow runs."""

import logging
import os
from pathlib import Path

import hydra
import mlflow
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from omegaconf import DictConfig

# Set project root
os.environ["PROJECT_ROOT"] = str(Path(__file__).parents[1].absolute())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Plot training metrics from MLflow runs.

    Args:
        cfg: Hydra config
    """
    # Set style
    sns.set_style(cfg.plotting.style)
    plt.figure(figsize=cfg.plotting.figure_size)

    # Get experiment
    experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {cfg.mlflow.experiment_name} not found")

    # Get metrics from all runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
    )

    if runs.empty:
        logger.warning("No runs found")
        return

    client = mlflow.tracking.MlflowClient()

    # Create separate plots for each metric group
    for metric_group, config in cfg.plotting.metrics.items():
        metrics_df = pd.DataFrame()

        for metric in config.metrics:
            metric_values = []
            steps = []
            for _, run_data in runs.iterrows():
                history = client.get_metric_history(run_data.run_id, metric)
                metric_values.extend([h.value for h in history])
                steps.extend([h.step for h in history])

            if metric_values:
                # Extract phase (train/val/test) from metric name
                phase = metric.split("_")[0]
                metrics_df = pd.concat(
                    [
                        metrics_df,
                        pd.DataFrame(
                            {
                                "step": steps,
                                "value": metric_values,
                                "phase": [phase] * len(steps),
                            }
                        ),
                    ],
                    ignore_index=True,
                )

        if not metrics_df.empty:
            plt.figure(figsize=cfg.plotting.figure_size)

            # Plot with custom styling
            for phase in metrics_df["phase"].unique():
                phase_data = metrics_df[metrics_df["phase"] == phase]
                plt.plot(
                    phase_data["step"],
                    phase_data["value"],
                    marker=cfg.plotting.markers[phase],
                    color=cfg.plotting.colors[phase],
                    label=phase,
                    alpha=0.8,
                )

            plt.title(f"{metric_group.capitalize()} During Training")
            plt.xlabel("Step")
            plt.ylabel(config.ylabel)
            plt.legend()

            # Save plot
            plot_dir = Path(cfg.paths.plots_dir) / "metrics"
            plot_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_dir / f"{metric_group}.png", dpi=cfg.plotting.dpi)
            plt.close()

            logger.info(f"Saved {metric_group} plot to {plot_dir}/{metric_group}.png")


if __name__ == "__main__":
    main()
