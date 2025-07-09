# -*- coding: utf-8 -*-
"""Visualization module for the Perceptron project.

This module contains the `Visualizer` class, which is responsible for
generating and logging all plots for a model training run. It is designed to
be used with Weights & Biases to log plots like confusion matrices, learning
curves, and decision boundaries, providing a comprehensive view of the model's
performance.

"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logging
from typing import List, Optional, Any
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
import wandb

# Handle both relative and absolute imports
try:
    from .constants import DECISION_BOUNDARY_RESOLUTION, DEFAULT_FIGURE_SIZE
except ImportError:
    from constants import DECISION_BOUNDARY_RESOLUTION, DEFAULT_FIGURE_SIZE


# --- Individual Plotting Functions ---

def _plot_decision_boundary(X: np.ndarray, y: np.ndarray, model: Any, class_names: Optional[List[str]]) -> Figure:
    """Creates a decision boundary plot for a trained 2D classifier.

    This version fills the regions with color to show the classification areas
    predicted by the model.

    Args:
        X: The input features (must be 2D) of shape (n_samples, 2).
        y: The true labels of shape (n_samples,).
        model: The trained Perceptron instance.
        class_names: Names of the classes for the legend.

    Returns:
        The figure object containing the plot.
    """
    if class_names is None:
        # Create default labels if none are provided
        unique_labels = np.unique(y)
        class_names = [f'Class {l}' for l in unique_labels]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    # Define a colormap for the regions and points
    cmap = ListedColormap(['#FF6347', '#4682B4'])  # Tomato, SteelBlue

    # Create a meshgrid to plot the decision boundary
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, DECISION_BOUNDARY_RESOLUTION),
                           np.arange(x2_min, x2_max, DECISION_BOUNDARY_RESOLUTION))

    # Predict the class for each point in the meshgrid
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # Plot the decision regions
    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    # Plot the original data points
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                   alpha=0.8, c=[cmap(idx)],
                   label=class_names[cl] if class_names and len(class_names) > cl else f'Class {cl}',
                   edgecolor='black')

    ax.set_title("Perceptron Decision Boundary")
    ax.set_xlabel("Input Feature 1")
    ax.set_ylabel("Input Feature 2")
    ax.legend(loc='upper left')
    ax.grid(True)
    return fig
    
def _plot_learning_curve(errors_per_epoch):
    """Creates a plot of the number of misclassifications per epoch.

    This helps visualize the model's learning progress over time. A downward
    trend indicates that the model is learning from the data.

    Args:
        errors_per_epoch (list[int]): A list containing the count of
                                      misclassifications for each epoch.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    fig, ax = plt.subplots()
    
    epochs = range(1, len(errors_per_epoch) + 1)
    ax.plot(epochs, errors_per_epoch, marker='o', linestyle='-',
            label='Misclassifications')

    ax.set_title("Perceptron Learning Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Number of Misclassifications")
    ax.legend()
    ax.grid(True)
    return fig


def _plot_confusion_matrix(y_true, y_pred, class_names):
    """Creates a heatmap plot of the confusion matrix.

    This plot provides a detailed breakdown of classification performance,
    showing correct predictions versus incorrect ones (false positives and
    false negatives). Uses seaborn for a visually appealing output.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.
        class_names (list[str]): Names of the classes for axis labels.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    if class_names is None:
        class_names = ['Class 0', 'Class 1']

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig


# --- Visualizer Class ---

class Visualizer:
    """A helper class to orchestrate the creation and logging of visualizations.

    This class centralizes all plotting logic and interaction with Weights & Biases,
    keeping the main training script clean and focused on the training loop.
    """

    def __init__(self, wandb_run, enabled=True):
        """
        Initializes the Visualizer instance.

        Args:
            wandb_run: The active `wandb` run object. This is used to log the plots.
            enabled (bool): If False, plotting and logging are disabled.
        """
        self.run = wandb_run
        self.enabled = enabled

    def _log_plot(self, plot_name, plot_fig):
        """Logs a matplotlib figure to Weights & Biases and closes it.

        Args:
            plot_name (str): The name to give the plot in the W&B dashboard.
            plot_fig (matplotlib.figure.Figure): The figure object to log.
        """
        if not self.enabled:
            plt.close(plot_fig)
            return

        logging.info(f"Logging '{plot_name}' to W&B.")
        self.run.log({plot_name: wandb.Image(plot_fig)})
        plt.close(plot_fig)

    def log_all(self, model, X, y, predictions, class_names):
        """
        Generates and logs all relevant plots for the model's performance.

        This is the main method to call after training. It generates a
        confusion matrix, a learning curve, and (if the data is 2D) a
        decision boundary plot.

        Args:
            model (Perceptron): The trained model instance.
            X (np.ndarray): The input features used for training.
            y (np.ndarray): The true labels.
            predictions (np.ndarray): The model's predictions on the input features.
            class_names (list[str]): The names of the classes for plot labels.
        """
        if not self.enabled:
            logging.info("Visualization is disabled.")
            return

        cm_fig = _plot_confusion_matrix(y, predictions, class_names=class_names)
        self._log_plot("Plots/Confusion_Matrix", cm_fig)

        learning_curve_fig = _plot_learning_curve(model.errors_per_epoch)
        self._log_plot("Plots/Learning_Curve", learning_curve_fig)

        if X.shape[1] == 2:
            boundary_fig = _plot_decision_boundary(X, y, model, class_names)
            self._log_plot("Plots/Decision_Boundary", boundary_fig)