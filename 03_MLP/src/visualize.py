# -*- coding: utf-8 -*-
"""Visualization module for the MLP project.

This module contains the `Visualizer` class, which is responsible for
generating and logging all plots for a model training run. It is designed to
be used with Weights & Biases to log plots like confusion matrices, loss
curves, and decision boundaries.
"""

import logging
from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def _log_predictions_table(
    X: np.ndarray, 
    y_true_labels: np.ndarray, 
    y_pred_labels: np.ndarray, 
    class_names: List[str], 
    num_images: int = 16
) -> wandb.Table:
    """Logs a wandb.Table with model predictions on a sample of images.
    
    Args:
        X: Input images of shape (n_samples, 784)
        y_true_labels: True class labels
        y_pred_labels: Predicted class labels
        class_names: List of class names
        num_images: Number of images to include in the table
        
    Returns:
        Wandb table with image predictions
    """
    table = wandb.Table(columns=["Image", "True Label", "Predicted Label"])

    # Take a random sample of images to log
    num_images = min(num_images, len(X))
    indices = np.random.choice(len(X), size=num_images, replace=False)

    for i in indices:
        # Reshape the flattened 784-pixel vector back to a 28x28 image
        image = X[i].reshape(28, 28)
        true_label = class_names[y_true_labels[i]] if y_true_labels[i] < len(class_names) else f"Class {y_true_labels[i]}"
        pred_label = class_names[y_pred_labels[i]] if y_pred_labels[i] < len(class_names) else f"Class {y_pred_labels[i]}"

        table.add_data(
            wandb.Image(image),
            true_label,
            pred_label
        )

    return table


def _plot_neuron_weights(W1: np.ndarray, num_neurons_to_show: int = 16) -> plt.Figure:
    """Visualizes the weights of the first N neurons in the hidden layer.
    
    Args:
        W1: Weight matrix of shape (784, hidden_size)
        num_neurons_to_show: Number of neurons to visualize
        
    Returns:
        Matplotlib figure object
        
    Raises:
        ValueError: If W1 doesn't have the expected shape for MNIST (784 inputs)
    """
    if W1.shape[0] != 784:
        raise ValueError(f"Expected 784 input features for MNIST, got {W1.shape[0]}")
    
    # Ensure we don't try to show more neurons than exist
    num_neurons_to_show = min(num_neurons_to_show, W1.shape[1])

    # Create a grid for the plots
    grid_size = int(np.ceil(np.sqrt(num_neurons_to_show)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i in range(num_neurons_to_show):
        # Get weights for the i-th neuron and reshape to 28x28
        neuron_weights = W1[:, i].reshape(28, 28)
        ax = axes[i]
        im = ax.imshow(neuron_weights, cmap='gray', interpolation='nearest')
        ax.set_title(f"Neuron {i+1}", fontsize=10)
        ax.axis('off')
        
        # Add colorbar for reference
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots
    for i in range(num_neurons_to_show, len(axes)):
        axes[i].axis('off')

    fig.suptitle('Hidden Layer Neuron Weights', fontsize=14)
    fig.tight_layout()
    return fig

# --- Individual Plotting Functions ---

def _plot_decision_boundary(
    X: np.ndarray, 
    y: np.ndarray, 
    model, 
    class_names: Optional[List[str]]
) -> plt.Figure:
    """Creates a decision boundary plot for a trained 2D classifier.
    
    Args:
        X: Input features of shape (n_samples, 2)
        y: True labels
        model: Trained model with predict method
        class_names: Optional list of class names
        
    Returns:
        Matplotlib figure object
        
    Raises:
        ValueError: If X doesn't have exactly 2 features
    """
    if X.shape[1] != 2:
        raise ValueError(f"Decision boundary plot requires 2D input, got {X.shape[1]}D")
    
    if class_names is None:
        unique_labels = np.unique(y)
        class_names = [f'Class {l}' for l in unique_labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    resolution = 0.02
    cmap = ListedColormap(['#FF6347', '#4682B4', '#32CD32', '#FFD700'])  # Support up to 4 classes

    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    # Handle one-hot encoded labels
    if y.ndim > 1 and y.shape[1] > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y.flatten()

    for idx, cl in enumerate(np.unique(y_labels)):
        ax.scatter(x=X[y_labels == cl, 0], y=X[y_labels == cl, 1],
                   alpha=0.8, c=[cmap(idx)],
                   label=class_names[cl] if class_names and len(class_names) > cl else f'Class {cl}',
                   edgecolor='black', s=50)

    ax.set_title("MLP Decision Boundary", fontsize=14)
    ax.set_xlabel("Input Feature 1", fontsize=12)
    ax.set_ylabel("Input Feature 2", fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    return fig
    return fig

def _plot_loss_curve(losses: List[float]) -> plt.Figure:
    """Creates a plot of the training loss per epoch.
    
    Args:
        losses: List of loss values per epoch
        
    Returns:
        Matplotlib figure object
    """
    if not losses:
        raise ValueError("Loss list is empty")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(losses) + 1)
    ax.plot(epochs, losses, marker='.', linestyle='-', linewidth=2, markersize=4)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('MLP Loss Curve', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add some statistics
    final_loss = losses[-1]
    min_loss = min(losses)
    ax.text(0.02, 0.98, f'Final Loss: {final_loss:.6f}\nMin Loss: {min_loss:.6f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    return fig

def _plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: Optional[List[str]]
) -> plt.Figure:
    """Creates a heatmap plot of the confusion matrix.
    
    Args:
        y_true: True labels (can be one-hot encoded)
        y_pred: Predicted labels
        class_names: Optional list of class names
        
    Returns:
        Matplotlib figure object
    """
    if class_names is None:
        n_classes = len(np.unique(y_true)) if y_true.ndim == 1 else y_true.shape[1]
        class_names = [f'Class {i}' for i in range(n_classes)]

    # If y_true is one-hot encoded, convert it to class indices
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true.flatten()

    # Ensure y_pred is 1D
    y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred

    cm = confusion_matrix(y_true_labels, y_pred_flat)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=ax1
    )
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14)
    
    # Plot percentages
    sns.heatmap(
        cm_percent, annot=True, fmt='.1f', cmap='Oranges',
        xticklabels=class_names, yticklabels=class_names, ax=ax2
    )
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=14)
    
    fig.tight_layout()
    return fig


class Visualizer:
    """A helper class to orchestrate the creation and logging of visualizations.
    
    This class manages the creation and logging of various plots including confusion
    matrices, loss curves, decision boundaries, and neuron weight visualizations.
    """

    def __init__(self, wandb_run, enabled: bool = True) -> None:
        """Initializes the Visualizer instance.
        
        Args:
            wandb_run: Active wandb run object for logging
            enabled: Whether visualization logging is enabled
        """
        self.run = wandb_run
        self.enabled = enabled
        logger.info(f"Visualizer initialized (enabled: {enabled})")

    def _log_plot(self, plot_name: str, plot_fig: plt.Figure) -> None:
        """Logs a matplotlib figure to Weights & Biases and closes it.
        
        Args:
            plot_name: Name for the plot in wandb
            plot_fig: Matplotlib figure to log
        """
        if not self.enabled:
            plt.close(plot_fig)
            return

        try:
            logger.info(f"Logging '{plot_name}' to W&B.")
            self.run.log({plot_name: wandb.Image(plot_fig, caption=plot_name)})
        except Exception as e:
            logger.error(f"Failed to log plot '{plot_name}': {e}")
        finally:
            plt.close(plot_fig)  # Always close to prevent memory leaks

    def log_all(
        self, 
        model, 
        X: np.ndarray, 
        y: np.ndarray, 
        predictions: np.ndarray, 
        class_names: Optional[List[str]]
    ) -> None:
        """Generates and logs all relevant plots for the model's performance.
        
        Args:
            model: Trained model with weights and predict method
            X: Input features for visualization
            y: True labels
            predictions: Model predictions
            class_names: Optional list of class names
        """
        if not self.enabled:
            logger.info("Visualization is disabled.")
            return

        logger.info("Generating and logging visualizations...")

        try:
            # Confusion Matrix
            cm_fig = _plot_confusion_matrix(y, predictions, class_names=class_names)
            self._log_plot("Plots/Confusion_Matrix", cm_fig)

            # Loss Curve
            if hasattr(model, 'losses') and model.losses:
                loss_curve_fig = _plot_loss_curve(model.losses)
                self._log_plot("Plots/Loss_Curve", loss_curve_fig)

            # Decision Boundary (only for 2D input)
            if X.shape[1] == 2:
                boundary_fig = _plot_decision_boundary(X, y, model, class_names)
                self._log_plot("Plots/Decision_Boundary", boundary_fig)

            # For MNIST (784 features), log a table of example predictions
            if X.shape[1] == 784:
                # Get 1D true labels from the potentially one-hot encoded y array
                if y.ndim > 1 and y.shape[1] > 1:
                    y_true_labels = np.argmax(y, axis=1)
                else:
                    y_true_labels = y.flatten()
                
                if class_names:
                    predictions_table = _log_predictions_table(
                        X, y_true_labels, predictions, class_names
                    )
                    self.run.log({"Predictions/Examples": predictions_table})

                # Also log the neuron weights
                if hasattr(model, 'W1'):
                    neuron_weights_fig = _plot_neuron_weights(model.W1)
                    self._log_plot("Parameters/Hidden_Neuron_Weights", neuron_weights_fig)

            logger.info("All visualizations logged successfully")

        except Exception as e:
            logger.error(f"Error during visualization: {e}")
            raise