# -*- coding: utf-8 -*-
"""Pure visualization functions for the Perceptron project.

This module contains plotting functions for creating visualizations of
perceptron model performance. These functions create matplotlib figures
that can be used by experiment tracking systems or displayed directly.

Following the separation of concerns principle:
- This module: Creates plots (what to visualize)
- wandb_integration.py: Handles experiment tracking (where to log)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logging
from typing import List, Optional, Any
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

# Handle both relative and absolute imports
try:
    from .constants import DECISION_BOUNDARY_RESOLUTION, DEFAULT_FIGURE_SIZE
except ImportError:
    from constants import DECISION_BOUNDARY_RESOLUTION, DEFAULT_FIGURE_SIZE


# --- Pure Visualization Functions ---

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: Optional[List[str]] = None) -> Figure:
    """Creates a confusion matrix plot.

    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels  
        class_names: Names of the classes for axis labels

    Returns:
        matplotlib.figure.Figure: The confusion matrix plot
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


def plot_learning_curve(errors_per_epoch: List[int]) -> Figure:
    """Creates a learning curve plot showing errors per epoch.

    Args:
        errors_per_epoch: List of error counts for each epoch

    Returns:
        matplotlib.figure.Figure: The learning curve plot
    """
    epochs = range(1, len(errors_per_epoch) + 1)
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    ax.plot(epochs, errors_per_epoch, marker='o', linestyle='-',
            label='Misclassifications')

    ax.set_title("Perceptron Learning Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Number of Misclassifications")
    ax.legend()
    ax.grid(True)
    return fig


def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray, 
                          class_names: Optional[List[str]] = None) -> Optional[Figure]:
    """Creates a decision boundary plot (only for 2D data).

    Args:
        model: Trained perceptron model with predict method
        X: Input features (must be 2D for visualization)
        y: True labels
        class_names: Names of the classes for legend

    Returns:
        matplotlib.figure.Figure or None: The decision boundary plot if data is 2D
    """
    if X.shape[1] != 2:
        logging.warning("Decision boundary plot only supported for 2D data")
        return None

    if class_names is None:
        class_names = ['Class 0', 'Class 1']

    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    
    # Create a mesh to plot the decision boundary
    h = DECISION_BOUNDARY_RESOLUTION
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    
    # Plot the data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='black')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Perceptron Decision Boundary')
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor='#FF0000', markersize=10, label=class_names[0]),
                      plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor='#00FF00', markersize=10, label=class_names[1])]
    ax.legend(handles=legend_elements)
    
    return fig


# --- Backward Compatibility Functions ---

def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          class_names: Optional[List[str]] = None) -> Figure:
    """Backward compatibility wrapper for plot_confusion_matrix."""
    return plot_confusion_matrix(y_true, y_pred, class_names)


def _plot_learning_curve(errors_per_epoch: List[int]) -> Figure:
    """Backward compatibility wrapper for plot_learning_curve."""
    return plot_learning_curve(errors_per_epoch)


def _plot_decision_boundary(X: np.ndarray, y: np.ndarray, model: Any, 
                           class_names: Optional[List[str]] = None) -> Optional[Figure]:
    """Backward compatibility wrapper for plot_decision_boundary."""
    return plot_decision_boundary(model, X, y, class_names)