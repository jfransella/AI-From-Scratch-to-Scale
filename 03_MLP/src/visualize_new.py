"""
MLP Visualizer using Shared Framework
===================================

This module provides Multi-Layer Perceptron specific visualizations using the shared
visualization framework. It demonstrates how to extend the BaseVisualizer for
deep learning model analysis while leveraging common components.

Key Features:
- Confusion matrix analysis with educational annotations
- Training curve visualization with optimization insights
- Decision boundary plotting for 2D classification problems
- Neuron weight visualization for feature detector analysis
- Prediction examples table for model interpretability
- MNIST-specific visualizations with mathematical context

Educational Focus:
- Neural network learning dynamics
- Feature detector interpretation
- Classification performance patterns
- Optimization behavior analysis
- Decision boundary formation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging

# Import shared visualization framework
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_from_scratch_shared.visualization import (
    BaseVisualizer,
    TrainingCurveVisualizer,
    DataDistributionVisualizer,
    add_mathematical_context,
    add_performance_insights,
    EducationalAnnotator
)

try:
    # Try relative imports first (when run as module)
    from .config import PLOTS_DIR
except ImportError:
    # Fall back to absolute imports (when run as script)
    from config import PLOTS_DIR

logger = logging.getLogger(__name__)


class MLPVisualizer(BaseVisualizer):
    """
    Specialized visualizer for Multi-Layer Perceptrons using shared framework.
    
    This class extends BaseVisualizer to provide MLP-specific visualization
    capabilities while maintaining consistency with the shared framework.
    
    Features:
    - Confusion matrix analysis with educational insights
    - Training dynamics visualization
    - Decision boundary plotting for 2D problems
    - Neuron weight visualization for feature analysis
    - MNIST-specific visualizations
    - Weights & Biases integration
    """
    
    def __init__(self, default_save_dir: Optional[Path] = None):
        """
        Initialize MLP visualizer.
        
        Args:
            default_save_dir: Default directory for saving plots
        """
        super().__init__(
            model_name="MLP",
            style_theme="educational",
            default_save_dir=default_save_dir or Path(PLOTS_DIR)
        )
        
        # Initialize shared component visualizers
        self.training_curve_viz = TrainingCurveVisualizer()
        self.data_viz = DataDistributionVisualizer()
        self.annotator = EducationalAnnotator(self.colors)
        
        # MLP-specific styling
        self.classification_colors = {
            'class_0': self.colors['primary'],
            'class_1': self.colors['secondary'], 
            'class_2': self.colors['accent'],
            'class_3': self.colors['error'],       # Changed from warning to error
            'boundary': self.colors['text'],
            'background': self.colors['background']
        }
        
        logger.debug("Initialized MLPVisualizer with shared framework")
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             title: str = "Confusion Matrix",
                             save_path: Optional[Union[str, Path]] = None,
                             show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create confusion matrix with educational annotations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names for display
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            Tuple of (figure, axes)
        """
        # Handle one-hot encoded labels
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = self.create_figure(figsize='confusion_matrix')
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names or [f'Class {i}' for i in range(cm.shape[1])],
            yticklabels=class_names or [f'Class {i}' for i in range(cm.shape[0])],
            ax=ax
        )
        
        # Styling
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        
        # Calculate performance metrics
        accuracy = np.trace(cm) / np.sum(cm)
        per_class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
        
        # Add performance insights
        insights = {
            "Overall Accuracy": float(accuracy),
            "Total Samples": float(np.sum(cm)),
            "Classes": float(cm.shape[0]),
            "Avg Class Accuracy": float(np.mean(per_class_accuracy))
        }
        
        interpretations = {
            "Overall Accuracy": "Fraction of correctly classified samples",
            "Total Samples": "Total number of test samples",
            "Classes": "Number of distinct classes in dataset",
            "Avg Class Accuracy": "Average per-class classification accuracy"
        }
        
        add_performance_insights(ax, insights, interpretations, position="bottom_right")
        
        # Add mathematical context
        add_mathematical_context(
            ax,
            concept="Classification Accuracy",
            formula=r"\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}",
            explanation="Diagonal elements represent correct classifications."
        )
        
        # Save and show
        self.save_and_show(fig, save_path=save_path, show=show)
        
        return fig, ax
    
    def plot_training_curves(self,
                           losses: List[float],
                           accuracies: Optional[List[float]] = None,
                           title: str = "Training Dynamics",
                           save_path: Optional[Union[str, Path]] = None,
                           show: bool = True) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """
        Plot training curves with educational annotations.
        
        Args:
            losses: Training loss values over epochs
            accuracies: Optional training accuracy values
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            Tuple of (figure, axes)
        """
        # Create subplots based on available data
        if accuracies is not None:
            fig, (ax1, ax2) = self.create_figure(figsize=(12, 5), subplots=(1, 2))
        else:
            fig, ax1 = self.create_figure(figsize='training_curves')
            ax2 = None
        
        # Plot loss curve
        epochs = range(1, len(losses) + 1)
        ax1.plot(epochs, losses, color=self.colors['primary'], linewidth=2, marker='o', markersize=3)
        ax1.set_title("Training Loss", fontweight='bold')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        
        # Add loss insights
        final_loss = losses[-1]
        min_loss = min(losses)
        loss_reduction = losses[0] - final_loss if len(losses) > 1 else 0
        
        self.add_educational_annotation(
            ax1,
            f"Final Loss: {final_loss:.4f}\n"
            f"Min Loss: {min_loss:.4f}\n"
            f"Reduction: {loss_reduction:.4f}",
            position="top_right"
        )
        
        # Plot accuracy curve if available
        if accuracies is not None and ax2 is not None:
            ax2.plot(epochs, accuracies, color=self.colors['success'], linewidth=2, marker='o', markersize=3)
            ax2.set_title("Training Accuracy", fontweight='bold')
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.grid(True, alpha=0.3)
            
            # Add accuracy insights
            final_acc = accuracies[-1]
            max_acc = max(accuracies)
            
            self.add_educational_annotation(
                ax2,
                f"Final Accuracy: {final_acc:.3f}\n"
                f"Max Accuracy: {max_acc:.3f}",
                position="bottom_right"
            )
        
        # Overall title
        if ax2 is not None:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        else:
            ax1.set_title(title, fontweight='bold', pad=20)
        
        # Add mathematical context
        if ax2 is not None:
            add_mathematical_context(
                ax1,
                concept="Cross-Entropy Loss",
                formula=r"L = -\frac{1}{N}\sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})",
                explanation="Lower loss indicates better fit to training data."
            )
        else:
            add_mathematical_context(
                ax1,
                concept="Gradient Descent Optimization",
                formula=r"\theta_{t+1} = \theta_t - \alpha \nabla_\theta L(\theta_t)",
                explanation="Loss should decrease as optimization progresses."
            )
        
        plt.tight_layout()
        
        # Save and show
        self.save_and_show(fig, save_path=save_path, show=show)
        
        return fig, (ax1, ax2) if ax2 is not None else ax1
    
    def plot_decision_boundary(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              model,
                              class_names: Optional[List[str]] = None,
                              title: str = "Decision Boundary",
                              save_path: Optional[Union[str, Path]] = None,
                              show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot decision boundary for 2D classification problems.
        
        Args:
            X: Input features (must be 2D)
            y: True labels
            model: Trained MLP model
            class_names: Optional class names
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            Tuple of (figure, axes)
        """
        if X.shape[1] != 2:
            raise ValueError(f"Decision boundary requires 2D input, got {X.shape[1]}D")
        
        # Handle one-hot encoded labels
        if y.ndim > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        
        # Create figure
        fig, ax = self.create_figure(figsize='decision_boundary')
        
        # Create mesh
        h = 0.02  # Step size
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Make predictions on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points)
        
        # Handle model output format
        if Z.ndim > 1 and Z.shape[1] > 1:
            Z = np.argmax(Z, axis=1)
        
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        n_classes = len(np.unique(y))
        colors = [self.classification_colors.get(f'class_{i}', self.colors['primary']) 
                 for i in range(n_classes)]
        
        ax.contourf(xx, yy, Z, alpha=0.3, colors=colors)
        
        # Plot data points
        for class_idx in range(n_classes):
            mask = (y == class_idx)
            if np.any(mask):
                label = class_names[class_idx] if class_names else f'Class {class_idx}'
                ax.scatter(X[mask, 0], X[mask, 1], 
                          c=colors[class_idx], 
                          label=label,
                          alpha=0.8,
                          edgecolors='black',
                          linewidth=1)
        
        # Styling
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add educational annotation
        self.add_educational_annotation(
            ax,
            f"Decision boundary shows learned classification regions.\n"
            f"Points represent training data ({X.shape[0]} samples).\n"
            f"Colors indicate predicted class regions.",
            position="top_right"
        )
        
        # Add mathematical context
        add_mathematical_context(
            ax,
            concept="Neural Network Decision Boundary",
            formula=r"f(x) = \sigma(W_2 \sigma(W_1 x + b_1) + b_2)",
            explanation="Non-linear boundaries formed by hidden layer transformations."
        )
        
        # Save and show
        self.save_and_show(fig, save_path=save_path, show=show)
        
        return fig, ax
    
    def plot_neuron_weights(self,
                           W1: np.ndarray,
                           num_neurons_to_show: int = 16,
                           title: str = "Hidden Layer Feature Detectors",
                           save_path: Optional[Union[str, Path]] = None,
                           show: bool = True) -> Tuple[plt.Figure, np.ndarray]:
        """
        Visualize hidden layer neuron weights as feature detectors.
        
        Args:
            W1: First layer weight matrix (input_size, hidden_size)
            num_neurons_to_show: Number of neurons to visualize
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            Tuple of (figure, axes array)
        """
        # Determine if this is MNIST format (784 = 28x28)
        input_size = W1.shape[0]
        if input_size == 784:
            img_shape = (28, 28)
        else:
            # Try to find a square image size
            img_size = int(np.sqrt(input_size))
            if img_size * img_size == input_size:
                img_shape = (img_size, img_size)
            else:
                raise ValueError(f"Cannot visualize weights for input size {input_size}")
        
        # Limit neurons to show
        hidden_size = W1.shape[1]
        num_neurons_to_show = min(num_neurons_to_show, hidden_size)
        
        # Calculate grid layout
        grid_size = int(np.ceil(np.sqrt(num_neurons_to_show)))
        
        # Create figure with subplots
        fig, axes = self.create_figure(
            figsize=(12, 12),
            subplots=(grid_size, grid_size)
        )
        
        # Ensure axes is always 2D array
        if grid_size == 1:
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape(1, -1)
        
        # Plot neuron weights
        for i in range(num_neurons_to_show):
            row = i // grid_size
            col = i % grid_size
            ax = axes[row, col]
            
            # Get weights for this neuron and reshape to image
            weights = W1[:, i].reshape(img_shape)
            
            # Normalize weights for visualization
            weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
            
            # Display weight pattern
            im = ax.imshow(weights_norm, cmap='RdBu_r', interpolation='nearest')
            ax.set_title(f'Neuron {i+1}', fontsize=10)
            ax.axis('off')
            
            # Add colorbar for first subplot
            if i == 0:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Weight Strength', fontsize=8)
        
        # Hide unused subplots
        for i in range(num_neurons_to_show, grid_size * grid_size):
            row = i // grid_size
            col = i % grid_size
            axes[row, col].axis('off')
        
        # Overall title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Add educational annotation
        fig.text(
            0.02, 0.02,
            f"Each subplot shows weights connecting input pixels to one hidden neuron.\n"
            f"These patterns represent learned feature detectors (edges, shapes, textures).\n"
            f"Showing {num_neurons_to_show} of {hidden_size} total hidden neurons.",
            bbox={
                'boxstyle': 'round,pad=0.5',
                'facecolor': self.colors['background'],
                'edgecolor': self.colors['primary'],
                'alpha': 0.9
            },
            fontsize=10,
            ha='left',
            va='bottom'
        )
        
        plt.tight_layout()
        
        # Save and show
        self.save_and_show(fig, save_path=save_path, show=show)
        
        return fig, axes
    
    def create_predictions_table(self,
                               X: np.ndarray,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               class_names: List[str],
                               num_samples: int = 10) -> wandb.Table:
        """
        Create Weights & Biases table with prediction examples.
        
        Args:
            X: Input features
            y_true: True labels  
            y_pred: Predicted labels
            class_names: List of class names
            num_samples: Number of samples to include
            
        Returns:
            Weights & Biases table object
        """
        # Handle one-hot encoded labels
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Sample indices
        indices = np.random.choice(len(X), size=min(num_samples, len(X)), replace=False)
        
        # Create table columns
        columns = ["Index", "Image", "True Label", "Predicted Label", "Correct"]
        data = []
        
        for idx in indices:
            # Convert image for W&B (assumes MNIST format)
            if X.shape[1] == 784:
                img = X[idx].reshape(28, 28)
                wandb_img = wandb.Image(img, caption=f"Sample {idx}")
            else:
                wandb_img = f"Feature vector {idx}"
            
            # Get labels
            true_label = class_names[y_true[idx]] if y_true[idx] < len(class_names) else f"Class {y_true[idx]}"
            pred_label = class_names[y_pred[idx]] if y_pred[idx] < len(class_names) else f"Class {y_pred[idx]}"
            is_correct = y_true[idx] == y_pred[idx]
            
            data.append([idx, wandb_img, true_label, pred_label, is_correct])
        
        return wandb.Table(columns=columns, data=data)


class Visualizer:
    """
    Legacy compatibility wrapper for W&B-integrated visualization.
    
    This class maintains the original API while leveraging the shared framework
    underneath for consistency across the project.
    """
    
    def __init__(self, wandb_run, enabled: bool = True):
        """
        Initialize legacy visualizer wrapper.
        
        Args:
            wandb_run: Active Weights & Biases run
            enabled: Whether visualization is enabled
        """
        self.run = wandb_run
        self.enabled = enabled
        self.visualizer = MLPVisualizer()
        
        logger.info(
            f"Legacy Visualizer initialized (enabled: {enabled})"
            f"{', W&B run: ' + wandb_run.name if enabled and hasattr(wandb_run, 'name') else ''}"
        )
    
    def _log_plot(self, plot_name: str, plot_fig: plt.Figure) -> None:
        """Log matplotlib figure to Weights & Biases."""
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
        finally:
            plt.close(plot_fig)
    
    def log_all(self,
               model,
               X: np.ndarray,
               y: np.ndarray,
               predictions: np.ndarray,
               class_names: Optional[List[str]] = None) -> None:
        """
        Generate and log all relevant visualizations using shared framework.
        
        Args:
            model: Trained MLP model
            X: Input features
            y: True labels
            predictions: Model predictions
            class_names: Optional class names
        """
        if not self.enabled:
            logger.info("Visualization logging is disabled - skipping all plots")
            return

        logger.info("=" * 60)
        logger.info("GENERATING EDUCATIONAL VISUALIZATIONS (Shared Framework)")
        logger.info("=" * 60)

        generated_plots = []
        
        try:
            # 1. Confusion Matrix
            logger.info("📊 Generating confusion matrix...")
            fig, _ = self.visualizer.plot_confusion_matrix(
                y, predictions, class_names=class_names, show=False
            )
            self._log_plot("Plots/Confusion_Matrix", fig)
            generated_plots.append("Confusion Matrix")

            # 2. Loss Curve
            if hasattr(model, 'losses') and model.losses:
                logger.info("📈 Generating loss curve...")
                accuracies = getattr(model, 'accuracies', None)
                fig, _ = self.visualizer.plot_training_curves(
                    model.losses, accuracies=accuracies, show=False
                )
                self._log_plot("Plots/Training_Curves", fig)
                generated_plots.append("Training Curves")
            else:
                logger.warning("⚠️  No loss history found - skipping training curves")

            # 3. Decision Boundary (2D only)
            if X.shape[1] == 2:
                logger.info("🎯 Generating decision boundary...")
                fig, _ = self.visualizer.plot_decision_boundary(
                    X, y, model, class_names, show=False
                )
                self._log_plot("Plots/Decision_Boundary", fig)
                generated_plots.append("Decision Boundary")
            else:
                logger.debug(f"Input has {X.shape[1]} features - skipping decision boundary")

            # 4. MNIST-specific visualizations
            if X.shape[1] == 784:
                logger.info("🖼️  Detected MNIST format - generating specialized visualizations...")
                
                # Prediction examples table
                if class_names:
                    try:
                        predictions_table = self.visualizer.create_predictions_table(
                            X, y, predictions, class_names
                        )
                        self.run.log({"Predictions/Examples": predictions_table})
                        generated_plots.append("Prediction Examples")
                        logger.info("✅ Generated prediction examples table")
                    except Exception as e:
                        logger.error(f"❌ Failed to generate prediction examples: {e}")

                # Neuron weights
                if hasattr(model, 'W1') and model.W1 is not None:
                    try:
                        logger.info("🧠 Visualizing hidden neuron weights...")
                        fig, _ = self.visualizer.plot_neuron_weights(model.W1, show=False)
                        self._log_plot("Parameters/Hidden_Neuron_Weights", fig)
                        generated_plots.append("Neuron Weights")
                        logger.info("✅ Generated neuron weight visualization")
                    except Exception as e:
                        logger.error(f"❌ Failed to generate neuron weights: {e}")

            # Summary
            logger.info("=" * 60)
            logger.info("VISUALIZATION SUMMARY (Shared Framework)")
            logger.info("=" * 60)
            logger.info(f"✅ Successfully generated {len(generated_plots)} visualizations:")
            for plot_name in generated_plots:
                logger.info(f"   • {plot_name}")
            
            if self.run and hasattr(self.run, 'url'):
                logger.info(f"🔗 View all plots at: {self.run.url}")
                
            logger.info("📚 Enhanced with shared framework features:")
            logger.info("   • Mathematical context annotations")
            logger.info("   • Performance insights")
            logger.info("   • Educational explanations")
            logger.info("   • Consistent styling across models")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"❌ Critical error during visualization generation: {e}")
            raise RuntimeError(f"Visualization pipeline failed: {e}") from e


# Wrapper functions for backwards compatibility
def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None,
                         show: bool = True) -> plt.Figure:
    """Wrapper for confusion matrix plotting."""
    viz = MLPVisualizer()
    fig, _ = viz.plot_confusion_matrix(y_true, y_pred, class_names, save_path=save_path, show=show)
    viz.cleanup_figures()
    return fig


def plot_learning_curve(losses: List[float],
                       accuracies: Optional[List[float]] = None,
                       save_path: Optional[str] = None,
                       show: bool = True) -> plt.Figure:
    """Wrapper for learning curve plotting."""
    viz = MLPVisualizer()
    fig, _ = viz.plot_training_curves(losses, accuracies, save_path=save_path, show=show)
    viz.cleanup_figures()
    return fig


def plot_decision_boundary(model,
                          X: np.ndarray,
                          y: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None,
                          show: bool = True) -> plt.Figure:
    """Wrapper for decision boundary plotting."""
    viz = MLPVisualizer()
    fig, _ = viz.plot_decision_boundary(X, y, model, class_names, save_path=save_path, show=show)
    viz.cleanup_figures()
    return fig


def plot_neuron_weights(W1: np.ndarray,
                       num_neurons_to_show: int = 16,
                       save_path: Optional[str] = None,
                       show: bool = True) -> plt.Figure:
    """Wrapper for neuron weight plotting."""
    viz = MLPVisualizer()
    fig, _ = viz.plot_neuron_weights(W1, num_neurons_to_show, save_path=save_path, show=show)
    viz.cleanup_figures()
    return fig


# Internal utility functions for legacy support
def _log_predictions_table(X: np.ndarray,
                          y_true_labels: np.ndarray,
                          predictions: np.ndarray,
                          class_names: List[str]) -> wandb.Table:
    """Legacy function for creating predictions table."""
    viz = MLPVisualizer()
    return viz.create_predictions_table(X, y_true_labels, predictions, class_names)


def _plot_neuron_weights(W1: np.ndarray, num_neurons_to_show: int = 16) -> plt.Figure:
    """Legacy function for neuron weight plotting."""
    return plot_neuron_weights(W1, num_neurons_to_show, show=False)


def _plot_decision_boundary(X: np.ndarray, y: np.ndarray, model, class_names: Optional[List[str]] = None) -> plt.Figure:
    """Legacy function for decision boundary plotting."""
    return plot_decision_boundary(model, X, y, class_names, show=False)


def _plot_loss_curve(losses: List[float]) -> plt.Figure:
    """Legacy function for loss curve plotting."""
    return plot_learning_curve(losses, show=False)


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None) -> plt.Figure:
    """Legacy function for confusion matrix plotting."""
    return plot_confusion_matrix(y_true, y_pred, class_names, show=False)
