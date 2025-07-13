"""
Common Visualization Components for Educational ML
=================================================

This module provides reusable visualization components that are commonly used
across different models in the AI-From-Scratch-to-Scale project. These components
ensure consistency and reduce code duplication while maintaining educational value.

Components:
- ConfusionMatrixVisualizer: Educational confusion matrices with metrics
- TrainingCurveVisualizer: Training/validation curves with analysis
- DecisionBoundaryVisualizer: 2D classification boundary plotting
- DataDistributionVisualizer: Dataset analysis and preprocessing plots

Each component is designed to be both standalone and easily integrated into
model-specific visualizers.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
import logging

from .base import BaseVisualizer
from .style import EDUCATIONAL_COLORS, CLASSIFICATION_COLORS, get_classification_colors
from .utils import save_and_show_plot, add_educational_annotation

logger = logging.getLogger(__name__)


class ConfusionMatrixVisualizer(BaseVisualizer):
    """
    Visualizer for educational confusion matrices with detailed statistics.
    
    This component creates confusion matrices optimized for learning, including:
    - Color-coded accuracy visualization
    - Per-class precision and recall statistics
    - Educational annotations explaining metrics
    - Both count and percentage displays
    """
    
    def __init__(self):
        super().__init__(model_name="ConfusionMatrix")
    
    def plot(self,
             y_true: np.ndarray,
             y_pred: np.ndarray,
             class_names: Optional[List[str]] = None,
             show_percentages: bool = True,
             show_statistics: bool = True,
             title: str = "Confusion Matrix",
             save_path: Optional[Union[str, Path]] = None,
             show: bool = True,
             xlabel: str = "Predicted Label",
             ylabel: str = "True Label") -> Tuple[plt.Figure, plt.Axes]:
        """
        Minimalist confusion matrix visualization: heatmap, axis labels, class names, colorbar, and a single small annotation.
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import numpy as np

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]

        # Generate class names if not provided
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]

        # Create figure
        fig, ax = self.create_figure(figsize='confusion_matrix')

        # Create heatmap
        if show_percentages:
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            annotations = np.array([
                [f"{count}\n({percent:.1f}%)"
                 for count, percent in zip(cm_row, percent_row)]
                for cm_row, percent_row in zip(cm, cm_percent)
            ])
        else:
            annotations = cm

        # Create heatmap with colorbar
        sns.heatmap(
            cm,
            annot=annotations,
            fmt='',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Number of Predictions'},
            square=True,
            annot_kws={"fontsize": 16, "fontweight": "bold"}
        )

        # Styling
        self.apply_consistent_styling(
            ax, title, xlabel, ylabel
        )

        # Minimal educational annotation (optional, small)
        # self.add_educational_annotation(
        #     ax,
        #     "Diagonal cells show correct predictions. Off-diagonal cells show classification errors.",
        #     position="bottom_right"
        # )

        # Save and show
        self.save_and_show(fig, save_path=save_path, show=show)
        return fig, ax
    
    def _add_statistics_annotation(self,
                                  cm: np.ndarray,
                                  ax: plt.Axes,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray) -> None:
        """Add statistical metrics annotation to confusion matrix."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Create statistics text
        stats_text = (
            f"Overall Metrics:\n"
            f"Accuracy: {accuracy:.3f}\n"
            f"Precision: {precision:.3f}\n"
            f"Recall: {recall:.3f}\n"
            f"F1-Score: {f1:.3f}"
        )
        
        # Add as annotation
        ax.text(
            1.02, 0.5,
            stats_text,
            transform=ax.transAxes,
            bbox={
                'boxstyle': 'round,pad=0.5',
                'facecolor': self.colors['background'],
                'edgecolor': self.colors['primary'],
                'alpha': 0.9
            },
            verticalalignment='center',
            fontsize=10
        )


class TrainingCurveVisualizer(BaseVisualizer):
    """
    Visualizer for training and validation curves with educational annotations.
    
    This component creates informative training progress visualizations including:
    - Loss and accuracy curves over time
    - Convergence analysis and annotations
    - Overfitting detection indicators
    - Learning rate effect visualization
    """
    
    def __init__(self):
        super().__init__(model_name="TrainingCurves")
    
    def plot_loss_curve(self,
                       train_losses: List[float],
                       val_losses: Optional[List[float]] = None,
                       title: str = "Training Loss Curve",
                       xlabel: str = "Epoch",
                       ylabel: str = "Loss",
                       show_convergence: bool = True,
                       save_path: Optional[Union[str, Path]] = None,
                       show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot training and validation loss curves.
        
        Args:
            train_losses: Training loss values
            val_losses: Validation loss values (optional)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show_convergence: Whether to add convergence analysis
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = self.create_figure(figsize='training_curves')
        
        epochs = range(1, len(train_losses) + 1)
        
        # Plot training loss
        ax.plot(
            epochs, train_losses,
            color=self.colors['primary'],
            linewidth=2,
            label='Training Loss',
            marker='o',
            markersize=4
        )
        
        # Plot validation loss if available
        if val_losses is not None:
            ax.plot(
                epochs, val_losses,
                color=self.colors['error'],
                linewidth=2,
                label='Validation Loss',
                marker='s',
                markersize=4
            )
            
            # Check for overfitting
            if show_convergence:
                self._analyze_overfitting(ax, train_losses, val_losses)
        
        # Add convergence analysis
        if show_convergence:
            self._add_convergence_analysis(ax, train_losses)
        
        # Styling
        self.apply_consistent_styling(ax, title, xlabel, ylabel)
        ax.legend()
        ax.set_yscale('log')  # Log scale often better for loss
        
        # Educational annotation
        self.add_educational_annotation(
            ax,
            "Loss should decrease over time.\n"
            "Validation loss plateau may indicate convergence.",
            position="top_right"
        )
        
        # Save and show
        self.save_and_show(fig, save_path=save_path, show=show)
        
        return fig, ax
    
    def plot_learning_curve(self,
                           errors_per_epoch: List[int],
                           title: str = "Learning Curve",
                           save_path: Optional[Union[str, Path]] = None,
                           show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot learning curve showing error count over epochs.
        
        Args:
            errors_per_epoch: Number of errors per epoch
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = self.create_figure(figsize='training_curves')
        
        epochs = range(1, len(errors_per_epoch) + 1)
        
        # Plot error curve
        ax.plot(
            epochs, errors_per_epoch,
            color=self.colors['error'],
            linewidth=2,
            marker='o',
            markersize=4
        )
        
        # Add convergence point if errors reach zero
        if min(errors_per_epoch) == 0:
            convergence_epoch = errors_per_epoch.index(0) + 1
            ax.axvline(
                convergence_epoch,
                color=self.colors['success'],
                linestyle='--',
                alpha=0.7,
                label=f'Convergence (Epoch {convergence_epoch})'
            )
            ax.legend()
        
        # Styling
        self.apply_consistent_styling(ax, title, "Epoch", "Number of Errors")
        
        # Educational annotation
        self.add_educational_annotation(
            ax,
            "Errors should decrease to zero for linearly separable data.\n"
            "Plateau indicates convergence or inseparable data.",
            position="top_right"
        )
        
        # Save and show
        self.save_and_show(fig, save_path=save_path, show=show)
        
        return fig, ax
    
    def _analyze_overfitting(self,
                            ax: plt.Axes,
                            train_losses: List[float],
                            val_losses: List[float]) -> None:
        """Analyze and annotate overfitting in training curves."""
        # Find potential overfitting point
        gap = np.array(val_losses) - np.array(train_losses)
        
        # Simple overfitting detection: where validation loss starts increasing
        # while training loss continues decreasing
        if len(val_losses) > 10:  # Need sufficient data
            val_trend = np.diff(val_losses[-10:])  # Last 10 epochs
            train_trend = np.diff(train_losses[-10:])
            
            if np.mean(val_trend) > 0 and np.mean(train_trend) < 0:
                # Potential overfitting
                ax.text(
                    0.7, 0.7,
                    "⚠️ Potential Overfitting Detected",
                    transform=ax.transAxes,
                    bbox={'boxstyle': 'round', 'facecolor': 'yellow', 'alpha': 0.7},
                    fontsize=9
                )
    
    def _add_convergence_analysis(self,
                                 ax: plt.Axes,
                                 losses: List[float]) -> None:
        """Add convergence analysis to training curve."""
        if len(losses) < 5:
            return
        
        # Check if converged (loss change < 1% over last 5 epochs)
        recent_losses = losses[-5:]
        if (max(recent_losses) - min(recent_losses)) / min(recent_losses) < 0.01:
            ax.text(
                0.02, 0.02,
                "✓ Converged",
                transform=ax.transAxes,
                bbox={'boxstyle': 'round', 'facecolor': 'lightgreen', 'alpha': 0.7},
                fontsize=9
            )


class DecisionBoundaryVisualizer(BaseVisualizer):
    """
    Visualizer for 2D classification decision boundaries.
    
    This component creates educational decision boundary plots including:
    - Mesh grid prediction visualization
    - Data point overlay with class colors
    - Boundary line highlighting
    - Educational annotations about decision regions
    """
    
    def __init__(self):
        super().__init__(model_name="DecisionBoundary")
    
    def plot(self,
             model: Any,
             X: np.ndarray,
             y: np.ndarray,
             class_names: Optional[List[str]] = None,
             resolution: float = 0.02,
             title: str = "Decision Boundary",
             save_path: Optional[Union[str, Path]] = None,
             show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create decision boundary visualization for 2D classification.
        
        Args:
            model: Trained model with predict() method
            X: Feature data (n_samples, 2)
            y: Labels (n_samples,)
            class_names: Names for classes
            resolution: Mesh grid resolution
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            Tuple of (figure, axes)
        """
        if X.shape[1] != 2:
            raise ValueError("Decision boundary visualization requires 2D data")
        
        fig, ax = self.create_figure(figsize='decision_boundary')
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, resolution),
            np.arange(y_min, y_max, resolution)
        )
        
        # Make predictions on mesh
        mesh_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        try:
            Z = model.predict(mesh_points)
            Z = Z.reshape(xx.shape)
        except Exception as e:
            logger.warning(f"Could not generate mesh predictions: {e}")
            Z = np.zeros_like(xx)
        
        # Get unique classes and colors
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        colors = get_classification_colors(n_classes)
        
        # Create colormap for decision regions
        cmap = ListedColormap(colors[:n_classes])
        
        # Plot decision regions
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=n_classes-1)
        
        # Plot decision boundary
        ax.contour(xx, yy, Z, colors='black', linewidths=2, linestyles='--', alpha=0.7)
        
        # Plot data points
        for i, class_val in enumerate(unique_classes):
            mask = y == class_val
            class_name = class_names[i] if class_names else f"Class {class_val}"
            
            ax.scatter(
                X[mask, 0], X[mask, 1],
                c=colors[i],
                marker='o',
                s=60,
                edgecolors='black',
                linewidth=1,
                label=class_name,
                alpha=0.8
            )
        
        # Styling
        self.apply_consistent_styling(ax, title, "Feature 1", "Feature 2")
        ax.legend()
        
        # Educational annotation
        self.add_educational_annotation(
            ax,
            "Colored regions show model predictions.\n"
            "Points show actual data with true labels.\n"
            "Dashed line shows decision boundary.",
            position="top_left"
        )
        
        # Save and show
        self.save_and_show(fig, save_path=save_path, show=show)
        
        return fig, ax


class DataDistributionVisualizer(BaseVisualizer):
    """
    Visualizer for dataset analysis and distribution plots.
    
    This component creates visualizations for understanding datasets including:
    - Feature distribution histograms
    - Class balance visualization
    - Correlation matrices
    - Data preprocessing before/after comparisons
    """
    
    def __init__(self):
        super().__init__(model_name="DataDistribution")
    
    def plot_feature_distributions(self,
                                  X: np.ndarray,
                                  y: Optional[np.ndarray] = None,
                                  feature_names: Optional[List[str]] = None,
                                  class_names: Optional[List[str]] = None,
                                  title: str = "Feature Distributions",
                                  save_path: Optional[Union[str, Path]] = None,
                                  show: bool = True) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot feature distribution histograms.
        
        Args:
            X: Feature data
            y: Labels (optional, for class-conditional distributions)
            feature_names: Names for features
            class_names: Names for classes
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            Tuple of (figure, axes array)
        """
        n_features = X.shape[1]
        
        # Calculate subplot layout
        cols = min(3, n_features)
        rows = (n_features + cols - 1) // cols
        
        fig, axes = self.create_figure(
            figsize=(4 * cols, 3 * rows),
            subplots=(rows, cols)
        )
        
        # Ensure axes is iterable
        if n_features == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Plot each feature
        for i in range(n_features):
            ax = axes[i]
            feature_name = feature_names[i] if feature_names else f"Feature {i+1}"
            
            if y is not None:
                # Class-conditional histograms
                unique_classes = np.unique(y)
                colors = get_classification_colors(len(unique_classes))
                
                for j, class_val in enumerate(unique_classes):
                    mask = y == class_val
                    class_name = class_names[j] if class_names else f"Class {class_val}"
                    
                    ax.hist(
                        X[mask, i],
                        bins=20,
                        alpha=0.6,
                        color=colors[j],
                        label=class_name,
                        density=True
                    )
                ax.legend()
            else:
                # Single histogram
                ax.hist(
                    X[:, i],
                    bins=20,
                    alpha=0.7,
                    color=self.colors['primary'],
                    density=True
                )
            
            ax.set_title(feature_name)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save and show
        self.save_and_show(fig, save_path=save_path, show=show)
        
        return fig, axes
    
    def plot_class_balance(self,
                          y: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          title: str = "Class Distribution",
                          save_path: Optional[Union[str, Path]] = None,
                          show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot class balance visualization.
        
        Args:
            y: Labels
            class_names: Names for classes
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = self.create_figure(figsize='default')
        
        # Count classes
        unique_classes, counts = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)
        
        # Get class names
        if class_names is None:
            class_names = [f"Class {i}" for i in unique_classes]
        
        # Get colors
        colors = get_classification_colors(n_classes)
        
        # Create bar plot
        bars = ax.bar(class_names, counts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + max(counts) * 0.01,
                str(count),
                ha='center',
                va='bottom'
            )
        
        # Styling
        self.apply_consistent_styling(ax, title, "Class", "Count")
        
        # Add balance analysis
        balance_ratio = min(counts) / max(counts)
        if balance_ratio < 0.5:
            balance_text = f"⚠️ Imbalanced dataset (ratio: {balance_ratio:.2f})"
            color = 'orange'
        else:
            balance_text = f"✓ Balanced dataset (ratio: {balance_ratio:.2f})"
            color = 'green'
        
        self.add_educational_annotation(
            ax, balance_text, position="top_right"
        )
        
        # Save and show
        self.save_and_show(fig, save_path=save_path, show=show)
        
        return fig, ax
