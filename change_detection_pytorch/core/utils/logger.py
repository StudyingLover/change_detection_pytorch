import os
import sys
import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


class Logger:
    """Logger that outputs to both terminal and log file, with plotting support.

    Creates run directory: runs/{task}/{timestamp}/ containing:
        - training.log: log file
        - metrics.json: all metrics history
        - curves.png: training/validation curves
    """

    def __init__(self, task: str, dataset: str, save_dir: str = './runs'):
        """
        Args:
            task: Model name (e.g., 'unet', 'casp')
            dataset: Dataset name (e.g., 'WHU-CD', 'LEVIR-CD')
            save_dir: Base directory for runs
        """
        self.task = task
        self.dataset = dataset
        self.base_dir = Path(save_dir)

        # Create run directory with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.base_dir / f'{task}_{dataset}_{timestamp}'
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Log file path
        self.log_file = self.run_dir / 'training.log'
        self.metrics_file = self.run_dir / 'metrics.json'

        # Metrics history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
        }

        # Initialize log file with header
        with open(self.log_file, 'w') as f:
            f.write(f"=== Training Log ===\n")
            f.write(f"Task: {task}\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Start Time: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Run Directory: {self.run_dir}\n")
            f.write(f"{'='*50}\n\n")

        self.print(f"Logger initialized: {self.run_dir}")

    def print(self, msg: str, print_to_terminal: bool = True):
        """Print message to both terminal and log file."""
        if print_to_terminal:
            print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')

    def log_epoch(self, epoch: int, train_logs: Dict, val_logs: Dict, lr: float):
        """Log one epoch's metrics.

        Args:
            epoch: Epoch number (0-indexed)
            train_logs: Dict with keys like 'cross_entropy_loss', 'iou_score', 'f_score', etc.
            val_logs: Same structure as train_logs
            lr: Current learning rate
        """
        # Extract loss and metrics
        train_loss = train_logs.get('cross_entropy_loss', 0)
        val_loss = val_logs.get('cross_entropy_loss', 0)
        train_iou = train_logs.get('iou_score', 0)
        val_iou = val_logs.get('iou_score', 0)
        train_f1 = train_logs.get('f_score', 0)
        val_f1 = val_logs.get('f_score', 0)

        # Build log message
        msg = (
            f"Epoch {epoch:3d} | "
            f"Train(L:{train_loss:.4f}, IoU:{train_iou:.4f}, F1:{train_f1:.4f}) | "
            f"Val(L:{val_loss:.4f}, IoU:{val_iou:.4f}, F1:{val_f1:.4f}) | "
            f"LR:{lr:.6f}"
        )
        self.print(msg)

        # Update history
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)

        # Dynamically add new metrics to history
        for key, value in train_logs.items():
            if key != 'cross_entropy_loss':
                train_key = f'train_{key}'
                val_key = f'val_{key}'
                if train_key not in self.history:
                    self.history[train_key] = []
                    self.history[val_key] = []
                self.history[train_key].append(value)
                self.history[val_key].append(val_logs.get(key, 0))

        # Save metrics to JSON (convert numpy types to Python native)
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return obj
        with open(self.metrics_file, 'w') as f:
            json.dump(convert_to_serializable(self.history), f, indent=2)

    def plot_curves(self, metrics_to_plot: Optional[List[str]] = None):
        """Plot training/validation curves and save to file.

        Args:
            metrics_to_plot: List of metric names to plot (without train_/val_ prefix).
                             If None, plots all available metrics.
        """
        if len(self.history['epoch']) == 0:
            self.print("No data to plot yet.")
            return

        # Determine which metrics to plot
        if metrics_to_plot is None:
            # Get all metric keys (excluding epoch, loss)
            all_keys = [k for k in self.history.keys() if k.startswith('train_') and k != 'train_loss']
            metrics_to_plot = [k.replace('train_', '') for k in all_keys]

        num_metrics = len(metrics_to_plot) + 1  # +1 for loss
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
        if num_metrics == 1:
            axes = [axes]

        epochs = self.history['epoch']

        # Plot loss
        ax = axes[0]
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot each metric
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i + 1]
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'

            if train_key in self.history and val_key in self.history:
                ax.plot(epochs, self.history[train_key], 'b-', label='Train', linewidth=2)
                ax.plot(epochs, self.history[val_key], 'r-', label='Val', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.set_title(metric.replace('_', ' ').title())
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        fig_path = self.run_dir / 'curves.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.print(f"Curves saved to: {fig_path}")

    def save_checkpoint_info(self, info: Dict):
        """Save checkpoint info to log."""
        with open(self.log_file, 'a') as f:
            f.write(f"\n--- Checkpoint Info ---\n")
            for k, v in info.items():
                f.write(f"{k}: {v}\n")
            f.write(f"{'-'*30}\n")

    @property
    def run_path(self) -> str:
        return str(self.run_dir)

    def finish(self):
        """Finalize logger: plot curves and write finish message."""
        self.print(f"\nTraining completed at {datetime.datetime.now().isoformat()}")
        self.print(f"Run directory: {self.run_dir}")
        self.plot_curves()


class TrainLogger:
    """Simplified logger for use within Epoch classes.

    This class provides a minimal interface for logging during training
    without requiring the full directory setup.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.current_epoch = 0
        self.history = {
            'train': {'loss': [], 'iou_score': [], 'f_score': [], 'precision': [], 'recall': []},
            'val': {'loss': [], 'iou_score': [], 'f_score': [], 'precision': [], 'recall': []}
        }

    def log_train(self, logs: Dict):
        """Log training metrics for current epoch."""
        for key, value in logs.items():
            if key in self.history['train']:
                self.history['train'][key].append(value)

    def log_val(self, logs: Dict):
        """Log validation metrics for current epoch."""
        for key, value in logs.items():
            if key in self.history['val']:
                self.history['val'][key].append(value)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def get_history(self) -> Dict:
        return self.history
