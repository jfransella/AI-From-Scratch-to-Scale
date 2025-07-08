# -*- coding: utf-8 -*-
"""Visualization module for the MLP project.

This module contains the `Visualizer` class, which is responsible for
generating and logging all plots for a model training run. It is designed to
be used with Weights & Biases to log plots like confusion matrices, loss
curves, and decision boundaries.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix


def _log_predictions_table(X, y_true_labels, y_pred_labels, class_names, num_images=16):
    """Logs a wandb.Table with model predictions on a sample of images."""
    table = wandb.Table(columns=["Image", "True Label", "Predicted Label"])

    # Take a random sample of images to log
    indices = np.random.choice(len(X), size=num_images, replace=False)

    for i in indices:
        # Reshape the flattened 784-pixel vector back to a 28x28 image
        image = X[i].reshape(28, 28)
        true_label = class_names[y_true_labels[i]]
        pred_label = class_names[y_pred_labels[i]]

        table.add_data(
            wandb.Image(image),
            true_label,
            pred_label
        )

    return table


def _plot_neuron_weights(W1, num_neurons_to_show=16):
    """Visualizes the weights of the first N neurons in the hidden layer."""
    # Ensure we don't try to show more neurons than exist
    num_neurons_to_show = min(num_neurons_to_show, W1.shape[1])

    # Create a grid for the plots
    grid_size = int(np.ceil(np.sqrt(num_neurons_to_show)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(num_neurons_to_show):
        # Get weights for the i-th neuron and reshape to 28x28
        neuron_weights = W1[:, i].reshape(28, 28)
        ax = axes[i]
        ax.imshow(neuron_weights, cmap='gray')
        ax.set_title(f"Neuron {i+1}")
        ax.axis('off')

    # Hide any unused subplots
    for i in range(num_neurons_to_show, len(axes)):
        axes[i].axis('off')

    fig.tight_layout()
    return fig

# --- Individual Plotting Functions ---

def _plot_decision_boundary(X, y, model, class_names):
    """Creates a decision boundary plot for a trained 2D classifier."""
    if class_names is None:
        unique_labels = np.unique(y)
        class_names = [f'Class {l}' for l in unique_labels]

    fig, ax = plt.subplots()
    resolution = 0.02
    cmap = ListedColormap(['#FF6347', '#4682B4'])  # Tomato, SteelBlue

    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y.flatten() == cl, 0], y=X[y.flatten() == cl, 1],
                   alpha=0.8, c=[cmap(idx)],
                   label=class_names[cl] if class_names and len(class_names) > cl else f'Class {cl}',
                   edgecolor='black')

    ax.set_title("MLP Decision Boundary")
    ax.set_xlabel("Input Feature 1")
    ax.set_ylabel("Input Feature 2")
    ax.legend(loc='upper left')
    ax.grid(True)
    return fig

def _plot_loss_curve(losses):
    """Creates a plot of the training loss per epoch."""
    fig, ax = plt.subplots()
    ax.plot(range(1, len(losses) + 1), losses, marker='.', linestyle='-')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Training Loss')
    ax.set_title('MLP Loss Curve')
    ax.grid(True)
    return fig

def _plot_confusion_matrix(y_true, y_pred, class_names):
    """Creates a heatmap plot of the confusion matrix."""
    if class_names is None:
        class_names = ['Class 0', 'Class 1']

    # If y_true is one-hot encoded, convert it to class indices
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true.flatten()

    cm = confusion_matrix(y_true_labels, y_pred.flatten())
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
    """A helper class to orchestrate the creation and logging of visualizations."""

    def __init__(self, wandb_run, enabled=True):
        """Initializes the Visualizer instance."""
        self.run = wandb_run
        self.enabled = enabled

    def _log_plot(self, plot_name, plot_fig):
        """Logs a matplotlib figure to Weights & Biases and closes it."""
        if not self.enabled:
            plt.close(plot_fig)
            return

        logging.info(f"Logging '{plot_name}' to W&B.")
        self.run.log({plot_name: wandb.Image(plot_fig)})
        plt.close(plot_fig)

    def log_all(self, model, X, y, predictions, class_names):
        """Generates and logs all relevant plots for the model's performance."""
        if not self.enabled:
            logging.info("Visualization is disabled.")
            return

        cm_fig = _plot_confusion_matrix(y, predictions, class_names=class_names)
        self._log_plot("Plots/Confusion_Matrix", cm_fig)

        loss_curve_fig = _plot_loss_curve(model.losses)
        self._log_plot("Plots/Loss_Curve", loss_curve_fig)

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
            predictions_table = _log_predictions_table(X, y_true_labels, predictions, class_names)
            self.run.log({"Predictions/Examples": predictions_table})

            # Also log the neuron weights
            neuron_weights_fig = _plot_neuron_weights(model.W1)
            self._log_plot("Parameters/Hidden_Neuron_Weights", neuron_weights_fig)