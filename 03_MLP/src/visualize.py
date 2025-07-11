# -*- coding: utf-8 -*-
"""Visualization module for the MLP project.

This module provides comprehensive visualization capabilities for Multi-Layer Perceptron
training and evaluation. It demonstrates key concepts in neural network interpretability:

Educational Focus:
    - Confusion matrices to understand classification performance patterns
    - Loss curves to visualize learning dynamics and convergence behavior
    - Decision boundaries to see how MLPs partition feature space (2D cases)
    - Neuron weight visualization to understand learned feature detectors
    - Prediction tables to inspect individual model predictions

The visualizations serve dual purposes:
    1. Educational: Help students understand what neural networks learn
    2. Practical: Enable experiment tracking and model debugging

Mathematical Context:
    - Decision boundaries show the hyperplane learned by the MLP
    - Neuron weights represent learned feature detectors (filters)
    - Loss curves demonstrate gradient descent optimization dynamics
    - Confusion matrices reveal classification decision patterns

Usage:
    The Visualizer class orchestrates all visualization creation and logging,
    designed to work seamlessly with Weights & Biases for experiment tracking.
"""

import logging
from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

# Configure logging for educational visibility
logger = logging.getLogger(__name__)

# Constants for visualization consistency
FIGURE_DPI = 300  # High resolution for documentation
DEFAULT_FIGURE_SIZE = (10, 6)
CONFUSION_MATRIX_SIZE = (15, 6)
NEURON_GRID_SIZE = (12, 12)
DEFAULT_COLORMAP = ['#FF6347', '#4682B4', '#32CD32', '#FFD700']  # Up to 4 classes


def _log_predictions_table(
    X: np.ndarray, 
    y_true_labels: np.ndarray, 
    y_pred_labels: np.ndarray, 
    class_names: List[str], 
    num_images: int = 16
) -> wandb.Table:
    """Creates a Weights & Biases table showing model predictions on sample images.
    
    Educational Purpose:
        This visualization helps students understand how their trained MLP
        makes individual predictions by showing actual images alongside
        their true and predicted labels.
    
    Mathematical Context:
        Each 784-dimensional input vector represents a flattened 28x28 MNIST image.
        The model's final softmax output determines the predicted class.
    
    Args:
        X: Input images of shape (n_samples, 784) - flattened MNIST images
        y_true_labels: True class labels (integers 0-9 for MNIST)
        y_pred_labels: Predicted class labels from model
        class_names: List of class names for display (e.g., ['0', '1', ..., '9'])
        num_images: Number of random sample images to include in table
        
    Returns:
        wandb.Table: Interactive table for Weights & Biases dashboard
        
    Raises:
        ValueError: If input shapes don't match MNIST format
    """
    # Validate input dimensions for educational clarity
    if X.shape[1] != 784:
        raise ValueError(
            f"Expected 784 features for MNIST images (28x28 flattened), "
            f"but got {X.shape[1]} features"
        )
    
    if len(y_true_labels) != len(y_pred_labels):
        raise ValueError(
            f"Mismatch between true labels ({len(y_true_labels)}) "
            f"and predictions ({len(y_pred_labels)})"
        )
    
    logger.debug(f"Creating predictions table with {num_images} sample images")
    
    # Initialize wandb table with descriptive columns
    table = wandb.Table(columns=["Image", "True Label", "Predicted Label", "Correct"])

    # Sample random images for visualization (reproducible sampling)
    num_images = min(num_images, len(X))
    np.random.seed(42)  # For reproducible visualization samples
    indices = np.random.choice(len(X), size=num_images, replace=False)

    for i in indices:
        # Educational step: Reshape flattened vector back to 2D image
        # This demonstrates the relationship between input format and visual data
        image = X[i].reshape(28, 28)
        
        # Handle edge cases where class indices might be out of bounds
        true_label_int = int(y_true_labels[i])  # Convert to int for indexing
        pred_label_int = int(y_pred_labels[i])  # Convert to int for indexing
        true_label = (
            class_names[true_label_int] 
            if true_label_int < len(class_names) 
            else f"Class {true_label_int}"
        )
        pred_label = (
            class_names[pred_label_int] 
            if pred_label_int < len(class_names) 
            else f"Class {pred_label_int}"
        )
        
        # Determine if prediction is correct for easy filtering in wandb
        is_correct = y_true_labels[i] == y_pred_labels[i]
        
        table.add_data(
            wandb.Image(image, caption=f"True: {true_label}, Pred: {pred_label}"),
            true_label,
            pred_label,
            "✓" if is_correct else "✗"
        )

    logger.debug(f"Successfully created predictions table with {len(indices)} images")
    return table


def _plot_neuron_weights(W1: np.ndarray, num_neurons_to_show: int = 16) -> plt.Figure:
    """Visualizes learned weight patterns of hidden layer neurons.
    
    Educational Purpose:
        This visualization reveals what feature detectors each neuron has learned.
        Students can see how individual neurons respond to different image patterns,
        providing insight into the internal representations of neural networks.
    
    Mathematical Context:
        Each neuron's weight vector W1[:, i] represents its learned feature detector.
        When reshaped to 28x28, these weights show what image patterns the neuron
        is sensitive to. Positive weights (bright) indicate features that activate
        the neuron, while negative weights (dark) indicate inhibitory patterns.
    
    Key Learning Points:
        - Early layers often learn edge and texture detectors
        - Each neuron specializes in different visual patterns
        - Weight magnitudes show feature importance
        - Random initialization leads to diverse learned features
    
    Args:
        W1: Weight matrix connecting input to hidden layer, shape (784, hidden_size)
            Each column W1[:, i] represents the weights for the i-th hidden neuron
        num_neurons_to_show: Number of neurons to visualize (default: 16)
        
    Returns:
        plt.Figure: Matplotlib figure with neuron weight visualizations
        
    Raises:
        ValueError: If W1 doesn't have 784 input features (MNIST format)
    """
    # Validate input dimensions for educational clarity
    if W1.shape[0] != 784:
        raise ValueError(
            f"Expected 784 input features for MNIST (28x28 flattened), "
            f"but got {W1.shape[0]} features. Cannot visualize as 28x28 images."
        )
    
    # Ensure we don't try to show more neurons than exist
    num_neurons_to_show = min(num_neurons_to_show, W1.shape[1])
    
    logger.debug(
        f"Visualizing weights for {num_neurons_to_show} neurons "
        f"out of {W1.shape[1]} total hidden neurons"
    )

    # Create a square grid layout for the neuron visualizations
    grid_size = int(np.ceil(np.sqrt(num_neurons_to_show)))
    fig, axes = plt.subplots(
        grid_size, grid_size, 
        figsize=NEURON_GRID_SIZE, 
        dpi=FIGURE_DPI
    )
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Statistical analysis for educational context
    weight_stats = {
        'mean_magnitude': np.mean(np.abs(W1)),
        'max_weight': np.max(W1),
        'min_weight': np.min(W1),
        'std_weight': np.std(W1)
    }

    for i in range(num_neurons_to_show):
        # Educational step: Reshape weight vector to image format
        # This shows how 1D weight vectors correspond to 2D image patterns
        neuron_weights = W1[:, i].reshape(28, 28)
        
        ax = axes[i]
        
        # Use diverging colormap to show positive/negative weights clearly
        im = ax.imshow(
            neuron_weights, 
            cmap='RdBu_r',  # Red-Blue diverging colormap
            interpolation='nearest',
            vmin=weight_stats['min_weight'],
            vmax=weight_stats['max_weight']
        )
        
        # Add neuron statistics for educational value
        neuron_magnitude = np.linalg.norm(neuron_weights)
        ax.set_title(
            f"Neuron {i+1}\n||w||₂ = {neuron_magnitude:.3f}", 
            fontsize=10
        )
        ax.axis('off')
        
        # Add colorbar for weight scale reference
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)

    # Hide any unused subplots for clean visualization
    for i in range(num_neurons_to_show, len(axes)):
        axes[i].axis('off')

    # Add comprehensive title with statistical information
    fig.suptitle(
        f'Hidden Layer Neuron Weights (Feature Detectors)\n'
        f'Mean |weight|: {weight_stats["mean_magnitude"]:.4f}, '
        f'Range: [{weight_stats["min_weight"]:.3f}, {weight_stats["max_weight"]:.3f}]',
        fontsize=14
    )
    fig.tight_layout()
    
    logger.debug("Successfully created neuron weights visualization")
    return fig

# --- Individual Plotting Functions ---

def _plot_decision_boundary(
    X: np.ndarray, 
    y: np.ndarray, 
    model, 
    class_names: Optional[List[str]]
) -> plt.Figure:
    """Creates a decision boundary visualization for 2D classification problems.
    
    Educational Purpose:
        This visualization shows how the MLP partitions the 2D feature space
        into different regions for each class. Students can see the non-linear
        decision boundaries that MLPs can learn, compared to linear classifiers.
    
    Mathematical Context:
        The decision boundary represents the set of points where the model's
        confidence between classes is equal. For MLPs, these boundaries can be
        highly non-linear due to the hidden layer transformations:
        
        Decision boundary: {x : P(class_i | x) = P(class_j | x) for i ≠ j}
        
        The visualization samples a dense grid of points and colors each region
        according to the model's predicted class.
    
    Key Learning Points:
        - MLPs can learn complex, non-linear decision boundaries
        - Hidden layers enable curved and disconnected decision regions
        - More hidden neurons generally enable more complex boundaries
        - Overfitting can create overly complex boundary patterns
    
    Args:
        X: Input features of shape (n_samples, 2) - must be 2D for visualization
        y: True labels, can be one-hot encoded or class indices
        model: Trained model with predict method
        class_names: Optional list of class names for legend
        
    Returns:
        plt.Figure: Matplotlib figure showing decision boundary and data points
        
    Raises:
        ValueError: If X doesn't have exactly 2 features
    """
    # Validate input dimensions for 2D visualization
    if X.shape[1] != 2:
        raise ValueError(
            f"Decision boundary visualization requires exactly 2 input features, "
            f"but got {X.shape[1]} features. Consider using dimensionality "
            f"reduction (PCA/t-SNE) for higher-dimensional data."
        )
    
    # Prepare class names for legend
    if class_names is None:
        unique_labels = np.unique(y)
        class_names = [f'Class {label}' for label in unique_labels]
    
    logger.debug(f"Creating decision boundary plot for {len(class_names)} classes")

    # Set up the plotting canvas
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE, dpi=FIGURE_DPI)
    
    # Define grid resolution (affects smoothness vs. computation time)
    resolution = 0.02  # Smaller values = smoother boundaries but slower computation
    
    # Use consistent colormap for educational consistency
    cmap = ListedColormap(DEFAULT_COLORMAP[:len(class_names)])
    
    # Create a mesh grid covering the feature space with some padding
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    
    # Educational step: Make predictions on grid points
    # This shows how the model classifies every point in the feature space
    grid_points = np.array([xx1.ravel(), xx2.ravel()]).T
    logger.debug(f"Making predictions on {len(grid_points)} grid points")
    
    try:
        Z = model.predict(grid_points)
        Z = Z.reshape(xx1.shape)
    except Exception as e:
        logger.error(f"Failed to generate decision boundary: {e}")
        raise RuntimeError(f"Model prediction failed during boundary generation: {e}")

    # Plot the decision boundary as colored regions
    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap, levels=len(class_names)-1)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    # Prepare labels for scatter plot (handle one-hot encoding)
    if y.ndim > 1 and y.shape[1] > 1:
        y_labels = np.argmax(y, axis=1)
        logger.debug("Converted one-hot encoded labels to class indices")
    else:
        y_labels = y.flatten()

    # Plot the actual data points with class-specific colors
    for idx, class_label in enumerate(np.unique(y_labels)):
        class_mask = y_labels == class_label
        class_label_int = int(class_label)  # Convert numpy scalar to int for indexing
        display_name = (
            class_names[class_label_int] 
            if class_label_int < len(class_names) 
            else f'Class {class_label_int}'
        )
        
        ax.scatter(
            X[class_mask, 0], X[class_mask, 1],
            alpha=0.8, 
            c=[cmap(idx)],
            label=display_name,
            edgecolor='black', 
            s=50,
            linewidth=0.5
        )

    # Add educational annotations and formatting
    ax.set_title(
        "MLP Decision Boundary\n(Non-linear classification regions)", 
        fontsize=14
    )
    ax.set_xlabel("Input Feature 1", fontsize=12)
    ax.set_ylabel("Input Feature 2", fontsize=12)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add text box with model information if available
    if hasattr(model, 'hidden_size'):
        info_text = f"Hidden neurons: {model.hidden_size}\nActivation: tanh"
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            fontsize=10
        )
    
    fig.tight_layout()
    logger.debug("Successfully created decision boundary visualization")
    return fig

def _plot_loss_curve(losses: List[float]) -> plt.Figure:
    """Creates a visualization of training loss progression over epochs.
    
    Educational Purpose:
        This plot is crucial for understanding neural network training dynamics.
        Students can observe convergence behavior, identify overfitting, and
        understand the effect of learning rates and optimization algorithms.
    
    Mathematical Context:
        The loss curve shows the progression of the cost function J(θ) over
        training iterations. For classification, this is typically:
        
        J(θ) = -1/m * Σᵢ Σⱼ yᵢⱼ * log(ŷᵢⱼ)  (cross-entropy loss)
        
        Where m is batch size, yᵢⱼ is true label, ŷᵢⱼ is predicted probability.
    
    Key Learning Points:
        - Steep initial decline shows rapid early learning
        - Plateau indicates convergence or learning rate too small
        - Oscillations may indicate learning rate too high
        - Smooth curve suggests good optimization dynamics
        - Final loss value indicates training quality
    
    Args:
        losses: List of loss values per epoch during training
        
    Returns:
        plt.Figure: Matplotlib figure with loss curve and statistics
        
    Raises:
        ValueError: If losses list is empty
    """
    # Validate input for educational clarity
    if not losses:
        raise ValueError(
            "Loss list is empty. Ensure model.fit() was called and "
            "loss values were recorded during training."
        )
    
    if len(losses) < 2:
        logger.warning(
            f"Only {len(losses)} loss value(s) available. "
            f"Consider training for more epochs for better visualization."
        )
    
    logger.debug(f"Creating loss curve with {len(losses)} data points")

    # Set up the plotting canvas
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE, dpi=FIGURE_DPI)
    epochs = range(1, len(losses) + 1)
    
    # Plot the loss curve with educational styling
    ax.plot(
        epochs, losses, 
        marker='.', 
        linestyle='-', 
        linewidth=2, 
        markersize=4,
        color='#2E86AB',  # Professional blue
        markerfacecolor='#A23B72',  # Contrasting marker color
        alpha=0.8
    )

    # Add educational annotations and formatting
    ax.set_xlabel('Training Epochs', fontsize=12)
    ax.set_ylabel('Training Loss (Cross-Entropy)', fontsize=12)
    ax.set_title(
        'MLP Training Dynamics: Loss Convergence\n'
        '(Lower values indicate better model fit)', 
        fontsize=14
    )
    ax.grid(True, alpha=0.3)
    
    # Calculate and display educational statistics
    final_loss = losses[-1]
    initial_loss = losses[0]
    min_loss = min(losses)
    min_loss_epoch = losses.index(min_loss) + 1
    
    # Calculate loss reduction percentage
    loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100
    
    # Add convergence analysis
    if len(losses) >= 10:
        # Check if loss is still decreasing significantly in last 10% of training
        recent_portion = max(1, len(losses) // 10)
        recent_losses = losses[-recent_portion:]
        recent_trend = "decreasing" if recent_losses[-1] < recent_losses[0] else "stable/increasing"
        
        convergence_text = (
            f"Initial Loss: {initial_loss:.6f}\n"
            f"Final Loss: {final_loss:.6f}\n"
            f"Best Loss: {min_loss:.6f} (epoch {min_loss_epoch})\n"
            f"Total Reduction: {loss_reduction:.1f}%\n"
            f"Recent Trend: {recent_trend}"
        )
    else:
        convergence_text = (
            f"Initial Loss: {initial_loss:.6f}\n"
            f"Final Loss: {final_loss:.6f}\n"
            f"Total Reduction: {loss_reduction:.1f}%"
        )
    
    # Add statistics box
    ax.text(
        0.02, 0.98, convergence_text,
        transform=ax.transAxes, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
        fontsize=10,
        family='monospace'
    )
    
    # Highlight minimum loss point if it's not the final point
    if min_loss_epoch != len(losses):
        ax.plot(
            min_loss_epoch, min_loss, 
            marker='*', 
            markersize=12, 
            color='red',
            label=f'Best Loss (Epoch {min_loss_epoch})'
        )
        ax.legend(loc='upper right')
    
    # Set y-axis to start from 0 or slightly below minimum for better visualization
    y_min = max(0, min_loss * 0.9)
    y_max = initial_loss * 1.1
    ax.set_ylim(y_min, y_max)
    
    fig.tight_layout()
    logger.debug("Successfully created loss curve visualization")
    return fig

def _plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: Optional[List[str]]
) -> plt.Figure:
    """Creates comprehensive confusion matrix visualizations for classification analysis.
    
    Educational Purpose:
        Confusion matrices reveal the detailed performance patterns of classifiers.
        Students can identify which classes are confused with each other and
        understand the types of errors their model makes.
    
    Mathematical Context:
        The confusion matrix C is defined as C[i,j] = number of samples with
        true class i predicted as class j. Key metrics derived from this:
        
        - Precision(i) = C[i,i] / Σⱼ C[j,i]  (true positives / total predicted as i)
        - Recall(i) = C[i,i] / Σⱼ C[i,j]     (true positives / total actual i)
        - Accuracy = Σᵢ C[i,i] / Σᵢⱼ C[i,j]   (correct predictions / total)
    
    Key Learning Points:
        - Diagonal elements represent correct predictions
        - Off-diagonal elements show confusion patterns
        - Symmetric confusion indicates balanced errors
        - Row-wise normalization shows recall (sensitivity)
        - Class imbalances appear as uneven row/column sums
    
    Args:
        y_true: True labels, can be one-hot encoded or class indices
        y_pred: Predicted labels from model
        class_names: Optional list of class names for axis labels
        
    Returns:
        plt.Figure: Matplotlib figure with count and percentage confusion matrices
        
    Raises:
        ValueError: If input arrays have incompatible shapes
    """
    # Validate inputs for educational clarity
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Mismatch between true labels ({len(y_true)}) "
            f"and predictions ({len(y_pred)})"
        )
    
    # Prepare class names for display
    if class_names is None:
        n_classes = len(np.unique(y_true)) if y_true.ndim == 1 else y_true.shape[1]
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    logger.debug(f"Creating confusion matrix for {len(class_names)} classes")

    # Handle one-hot encoded true labels
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_labels = np.argmax(y_true, axis=1)
        logger.debug("Converted one-hot encoded true labels to class indices")
    else:
        y_true_labels = y_true.flatten()

    # Ensure predictions are 1D
    y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred

    # Calculate confusion matrix
    try:
        cm = confusion_matrix(y_true_labels, y_pred_flat)
    except Exception as e:
        logger.error(f"Failed to compute confusion matrix: {e}")
        raise RuntimeError(f"Confusion matrix computation failed: {e}")
    
    # Calculate row-wise percentages (recall for each class)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm_percent = np.nan_to_num(cm_percent)  # Handle division by zero
    
    # Set up the plotting canvas with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=CONFUSION_MATRIX_SIZE, dpi=FIGURE_DPI)
    
    # Plot 1: Raw counts confusion matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names, 
        yticklabels=class_names, 
        ax=ax1,
        cbar_kws={'label': 'Sample Count'},
        square=True
    )
    ax1.set_xlabel('Predicted Class', fontsize=12)
    ax1.set_ylabel('True Class', fontsize=12)
    ax1.set_title('Confusion Matrix (Counts)\nDiagonal = Correct Predictions', fontsize=14)
    
    # Add total samples information
    total_samples = np.sum(cm)
    correct_predictions = np.trace(cm)
    overall_accuracy = correct_predictions / total_samples * 100
    
    ax1.text(
        0.02, 0.98, 
        f'Total Samples: {total_samples}\n'
        f'Correct: {correct_predictions}\n'
        f'Accuracy: {overall_accuracy:.1f}%',
        transform=ax1.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
        fontsize=10
    )
    
    # Plot 2: Percentage confusion matrix (row-normalized)
    sns.heatmap(
        cm_percent, 
        annot=True, 
        fmt='.1f', 
        cmap='Oranges',
        xticklabels=class_names, 
        yticklabels=class_names, 
        ax=ax2,
        cbar_kws={'label': 'Percentage (%)'},
        square=True
    )
    ax2.set_xlabel('Predicted Class', fontsize=12)
    ax2.set_ylabel('True Class', fontsize=12)
    ax2.set_title('Confusion Matrix (Row-wise %)\nShows Recall per Class', fontsize=14)
    
    # Calculate and display per-class recall
    class_recalls = np.diag(cm_percent)
    worst_class = int(np.argmin(class_recalls))  # Convert to int for indexing
    best_class = int(np.argmax(class_recalls))   # Convert to int for indexing
    
    recall_text = (
        f'Best Recall: {class_names[best_class]} ({class_recalls[best_class]:.1f}%)\n'
        f'Worst Recall: {class_names[worst_class]} ({class_recalls[worst_class]:.1f}%)\n'
        f'Mean Recall: {np.mean(class_recalls):.1f}%'
    )
    
    ax2.text(
        0.02, 0.98, recall_text,
        transform=ax2.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
        fontsize=10
    )
    
    # Adjust layout for readability
    fig.tight_layout()
    
    # Log educational insights
    logger.info(f"Confusion matrix created - Overall accuracy: {overall_accuracy:.1f}%")
    logger.debug(f"Per-class recall range: {np.min(class_recalls):.1f}% - {np.max(class_recalls):.1f}%")
    
    return fig


# --- Public Visualization Functions (Clean API) ---

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: Optional[List[str]] = None) -> plt.Figure:
    """Creates a confusion matrix plot.

    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels  
        class_names: Names of the classes for axis labels

    Returns:
        matplotlib.figure.Figure: The confusion matrix plot
    """
    return _plot_confusion_matrix(y_true, y_pred, class_names)


def plot_learning_curve(losses: List[float]) -> plt.Figure:
    """Creates a learning curve plot showing loss over epochs.

    Args:
        losses: List of loss values for each epoch

    Returns:
        matplotlib.figure.Figure: The learning curve plot
    """
    return _plot_loss_curve(losses)


def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray, 
                          class_names: Optional[List[str]] = None) -> Optional[plt.Figure]:
    """Creates a decision boundary plot (only for 2D data).

    Args:
        model: Trained MLP model with predict method
        X: Input features (must be 2D for visualization)
        y: True labels
        class_names: Names of the classes for legend

    Returns:
        matplotlib.figure.Figure or None: The decision boundary plot if data is 2D
    """
    if X.shape[1] != 2:
        logger.warning("Decision boundary plot only supported for 2D data")
        return None
    
    return _plot_decision_boundary(X, y, model, class_names)


def plot_neuron_weights(W1: np.ndarray, num_neurons_to_show: int = 16) -> plt.Figure:
    """Creates a visualization of hidden layer neuron weights.

    Args:
        W1: Hidden layer weight matrix
        num_neurons_to_show: Number of neurons to visualize

    Returns:
        matplotlib.figure.Figure: The neuron weights plot
    """
    return _plot_neuron_weights(W1, num_neurons_to_show)


class Visualizer:
    """Educational visualization orchestrator for MLP training and evaluation.
    
    This class manages the creation and logging of educational visualizations that
    help students understand neural network behavior, training dynamics, and
    model performance patterns.
    
    Educational Philosophy:
        Each visualization serves a specific learning objective:
        - Confusion matrices: Understanding classification performance patterns
        - Loss curves: Observing optimization and convergence behavior  
        - Decision boundaries: Visualizing learned decision regions (2D cases)
        - Neuron weights: Interpreting learned feature detectors
        - Prediction samples: Connecting model outputs to actual examples
    
    Integration with Weights & Biases:
        All visualizations are automatically logged to W&B for experiment tracking,
        enabling systematic comparison of different model configurations and
        hyperparameter settings.
    
    Attributes:
        run: Active Weights & Biases run for logging
        enabled: Flag to enable/disable visualization logging
    """

    def __init__(self, wandb_run, enabled: bool = True) -> None:
        """Initializes the Visualizer for educational experiment tracking.
        
        Args:
            wandb_run: Active Weights & Biases run object for logging visualizations
            enabled: Whether to generate and log visualizations (useful for debugging)
            
        Raises:
            ValueError: If wandb_run is None when enabled=True
        """
        if enabled and wandb_run is None:
            raise ValueError(
                "wandb_run cannot be None when visualization is enabled. "
                "Initialize wandb.init() first or set enabled=False."
            )
        
        self.run = wandb_run
        self.enabled = enabled
        
        logger.info(
            f"Visualizer initialized (enabled: {enabled})"
            f"{', W&B run: ' + wandb_run.name if enabled and hasattr(wandb_run, 'name') else ''}"
        )

    def _log_plot(self, plot_name: str, plot_fig: plt.Figure) -> None:
        """Safely logs a matplotlib figure to Weights & Biases with error handling.
        
        Educational Purpose:
            This method demonstrates proper resource management in visualization
            pipelines, ensuring figures are always closed to prevent memory leaks.
        
        Args:
            plot_name: Descriptive name for the plot in W&B dashboard
            plot_fig: Matplotlib figure object to log
            
        Note:
            The figure is always closed after logging to prevent memory accumulation
            during long training runs.
        """
        if not self.enabled:
            plt.close(plot_fig)
            logger.debug(f"Skipped logging '{plot_name}' (visualization disabled)")
            return

        try:
            logger.debug(f"Logging '{plot_name}' to Weights & Biases")
            self.run.log({plot_name: wandb.Image(plot_fig, caption=plot_name)})
            logger.debug(f"Successfully logged '{plot_name}'")
        except Exception as e:
            logger.error(f"Failed to log plot '{plot_name}': {e}")
            # Don't re-raise to avoid breaking the training pipeline
        finally:
            plt.close(plot_fig)  # Critical: Always close to prevent memory leaks

    def log_all(
        self, 
        model, 
        X: np.ndarray, 
        y: np.ndarray, 
        predictions: np.ndarray, 
        class_names: Optional[List[str]] = None
    ) -> None:
        """Generates and logs all relevant educational visualizations for the model.
        
        Educational Purpose:
            This method demonstrates a systematic approach to model evaluation
            through visualization. Students learn to examine their models from
            multiple perspectives: performance metrics, learning dynamics,
            decision boundaries, and internal representations.
        
        Visualization Strategy:
            1. Always generate confusion matrix (universal for classification)
            2. Show loss curve if training history is available
            3. Create decision boundary for 2D problems (toy datasets)
            4. Display neuron weights for MNIST (feature detector analysis)
            5. Show prediction examples for interpretability
        
        Args:
            model: Trained MLP model with weights and methods
            X: Input features used for evaluation
            y: True labels (can be one-hot encoded)
            predictions: Model predictions on X
            class_names: Optional class names for better visualization labels
            
        Raises:
            RuntimeError: If critical visualization generation fails
            
        Note:
            Individual visualization failures don't stop the entire process,
            ensuring robustness during experimentation.
        """
        if not self.enabled:
            logger.info("Visualization logging is disabled - skipping all plots")
            return

        logger.info("=" * 60)
        logger.info("GENERATING EDUCATIONAL VISUALIZATIONS")
        logger.info("=" * 60)

        # Track successful visualizations for summary
        generated_plots = []
        
        try:
            # 1. Confusion Matrix (Universal for classification)
            logger.info("📊 Generating confusion matrix...")
            cm_fig = _plot_confusion_matrix(y, predictions, class_names=class_names)
            self._log_plot("Plots/Confusion_Matrix", cm_fig)
            generated_plots.append("Confusion Matrix")

            # 2. Loss Curve (If training history available)
            if hasattr(model, 'losses') and model.losses:
                logger.info("📈 Generating loss curve...")
                loss_curve_fig = _plot_loss_curve(model.losses)
                self._log_plot("Plots/Loss_Curve", loss_curve_fig)
                generated_plots.append("Loss Curve")
            else:
                logger.warning("⚠️  No loss history found - skipping loss curve")

            # 3. Decision Boundary (Only for 2D toy problems)
            if X.shape[1] == 2:
                logger.info("🎯 Generating decision boundary...")
                boundary_fig = _plot_decision_boundary(X, y, model, class_names)
                self._log_plot("Plots/Decision_Boundary", boundary_fig)
                generated_plots.append("Decision Boundary")
            else:
                logger.debug(f"Input has {X.shape[1]} features - skipping decision boundary (requires 2D)")

            # 4. MNIST-specific visualizations
            if X.shape[1] == 784:  # MNIST format detection
                logger.info("🖼️  Detected MNIST format - generating specialized visualizations...")
                
                # 4a. Prediction examples table
                try:
                    # Handle one-hot encoded labels
                    if y.ndim > 1 and y.shape[1] > 1:
                        y_true_labels = np.argmax(y, axis=1)
                    else:
                        y_true_labels = y.flatten()
                    
                    if class_names:
                        predictions_table = _log_predictions_table(
                            X, y_true_labels, predictions, class_names
                        )
                        self.run.log({"Predictions/Examples": predictions_table})
                        generated_plots.append("Prediction Examples")
                        logger.info("✅ Generated prediction examples table")
                    else:
                        logger.warning("⚠️  No class names provided - skipping prediction table")
                
                except Exception as e:
                    logger.error(f"❌ Failed to generate prediction examples: {e}")

                # 4b. Neuron weight visualization
                if hasattr(model, 'W1') and model.W1 is not None:
                    try:
                        logger.info("🧠 Visualizing hidden neuron weights (feature detectors)...")
                        neuron_weights_fig = _plot_neuron_weights(model.W1)
                        self._log_plot("Parameters/Hidden_Neuron_Weights", neuron_weights_fig)
                        generated_plots.append("Neuron Weights")
                        logger.info("✅ Generated neuron weight visualization")
                    except Exception as e:
                        logger.error(f"❌ Failed to generate neuron weights: {e}")
                else:
                    logger.warning("⚠️  No hidden layer weights found - skipping neuron visualization")

            # Summary of generated visualizations
            logger.info("=" * 60)
            logger.info("VISUALIZATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"✅ Successfully generated {len(generated_plots)} visualizations:")
            for plot_name in generated_plots:
                logger.info(f"   • {plot_name}")
            
            if self.run and hasattr(self.run, 'url'):
                logger.info(f"🔗 View all plots at: {self.run.url}")
            
            logger.info("📚 Educational Value: Review each visualization to understand:")
            logger.info("   • Model performance patterns (confusion matrix)")
            logger.info("   • Learning dynamics (loss curve)")
            logger.info("   • Decision boundaries (if 2D)")
            logger.info("   • Feature detectors (neuron weights)")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"❌ Critical error during visualization generation: {e}")
            raise RuntimeError(f"Visualization pipeline failed: {e}") from e