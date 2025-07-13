"""
Plot Factory for Standardized Visualization Creation
==================================================

This module provides a PlotFactory class that creates standardized plots
for common visualization types across all models in the AI-From-Scratch-to-Scale project.

Key Features:
- Standardized plot creation for common visualization types
- Automatic figure sizing based on plot type
- Consistent styling application
- Built-in W&B logging integration
- Educational annotations and context

Educational Focus:
- Demonstrates factory pattern for software design
- Shows how to create reusable visualization components
- Provides consistent interfaces across different model types
- Enables systematic comparison of visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.figure as Figure
from matplotlib.axes import Axes
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import logging

from .base import BaseVisualizer
from .style import FIGURE_SIZES, FONT_SIZES, EDUCATIONAL_COLORS
from .utils import format_axes_for_education
from .validation import VisualizationValidator, ValidationError
from .performance import (
    PerformanceMonitor,
    FigureCache,
    LazyPlotCreator,
    MemoryManager,
    performance_monitor,
    lazy_plot_creation
)

logger = logging.getLogger(__name__)


class PlotFactory:
    """
    Factory class for creating standardized plots across all models.
    
    This class provides methods to create common visualization types
    with consistent styling, sizing, and educational annotations.
    
    Features:
    - Automatic figure sizing based on plot type
    - Consistent styling application
    - Built-in W&B logging integration
    - Educational annotations and context
    - Support for both single plots and multi-plot layouts
    
    Example:
        factory = PlotFactory(model_name="Perceptron")
        fig, ax = factory.create_training_plot(errors_per_epoch)
        factory.save_and_log(fig, "training_curve", plot_type="learning_curve")
    """
    
    def __init__(self, 
                 model_name: str,
                 wandb_visualizer: Optional[Any] = None,
                 save_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the plot factory.
        
        Args:
            model_name: Name of the model (e.g., "Perceptron", "MLP")
            wandb_visualizer: Optional W&B visualizer for logging
            save_dir: Directory for saving plots
        """
        self.model_name = model_name
        self.wandb_visualizer = wandb_visualizer
        self.save_dir = Path(save_dir) if save_dir else Path("outputs/plots")
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize base visualizer for common functionality
        self.base_visualizer = BaseVisualizer(
            model_name=model_name,
            default_save_dir=self.save_dir
        )
        
        logger.info(f"PlotFactory initialized for {model_name}")
        
        # Initialize performance optimization components
        self.performance_monitor = PerformanceMonitor()
        self.figure_cache = FigureCache()
        self.lazy_creator = LazyPlotCreator(self.figure_cache)
        self.memory_manager = MemoryManager()
    
    def _validate_training_data(self, training_data: Dict[str, List[float]]) -> None:
        """
        Validate training data for plot creation.
        
        Args:
            training_data: Training data dictionary
            
        Raises:
            ValidationError: If training data is invalid
        """
        validator = VisualizationValidator()
        validator.validate_training_data(training_data)
    
    def _validate_model_for_decision_boundary(self, model: Any, features: np.ndarray) -> None:
        """
        Validate model and features for decision boundary plot.
        
        Args:
            model: Model object
            features: Input features
            
        Raises:
            ValidationError: If model or features are invalid
        """
        validator = VisualizationValidator()
        
        # Validate model interface
        validator.validate_model_interface(model)
        
        # Validate features are 2D
        validator.validate_2d_features(features, "features")
    
    def _validate_confusion_matrix_inputs(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Validate inputs for confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Raises:
            ValidationError: If inputs are invalid
        """
        validator = VisualizationValidator()
        
        # Validate labels
        validator.validate_labels(y_true, data_name="y_true")
        validator.validate_labels(y_pred, data_name="y_pred")
        
        # Validate matching lengths
        if len(y_true) != len(y_pred):
            raise ValidationError(
                f"y_true and y_pred must have same length: {len(y_true)} vs {len(y_pred)}",
                [
                    "Check prediction generation",
                    "Ensure y_true and y_pred are aligned",
                    "Verify data preprocessing"
                ]
            )
    
    def create_training_plot(self, 
                           training_data: Dict[str, List[float]],
                           plot_type: str = "learning_curve",
                           title: Optional[str] = None,
                           **kwargs) -> Tuple[Figure.Figure, Axes]:
        """
        Create a standardized training plot (learning curve, loss curve, etc.).
        
        Args:
            training_data: Dictionary with training metrics (e.g., {'loss': [...], 'accuracy': [...]})
            plot_type: Type of training plot ('learning_curve', 'loss_curve', 'accuracy_curve')
            title: Optional custom title
            **kwargs: Additional plotting parameters
            
        Returns:
            Tuple of (figure, axes)
        """
        # Validate training data
        self._validate_training_data(training_data)
        
        # Determine figure size based on plot type
        figsize = FIGURE_SIZES.get(f'{plot_type}_training', FIGURE_SIZES['training_curves'])
        
        # Create figure with educational styling
        fig, ax = self.base_visualizer.create_figure(figsize=figsize)
        
        # Plot training data
        first_metric_values = next(iter(training_data.values()))
        epochs = np.arange(1, len(first_metric_values) + 1)
        
        for metric_name, values in training_data.items():
            ax.plot(epochs, values, 
                   label=metric_name.replace('_', ' ').title(),
                   linewidth=2, marker='o', markersize=4, alpha=0.8)
        
        # Apply consistent styling
        title = title or f"{self.model_name} Training Progress"
        self.base_visualizer.apply_consistent_styling(
            ax=ax,
            title=title,
            xlabel="Epoch",
            ylabel="Metric Value",
            grid=True
        )
        
        # Add educational annotation
        self._add_training_annotation(ax, plot_type)
        
        return fig, ax
    
    def create_decision_boundary(self, 
                               model: Any,
                               features: np.ndarray,
                               labels: np.ndarray,
                               resolution: float = 0.01,
                               title: Optional[str] = None,
                               **kwargs) -> Tuple[Figure.Figure, Axes]:
        """
        Create a standardized decision boundary visualization.
        
        Args:
            model: Trained model with predict method
            features: Input features (must be 2D for visualization)
            labels: True labels
            resolution: Mesh resolution for boundary
            title: Optional custom title
            **kwargs: Additional plotting parameters
            
        Returns:
            Tuple of (figure, axes)
        """
        # Validate inputs
        self._validate_model_for_decision_boundary(model, features)
        
        if features.shape[1] != 2:
            raise ValueError("Decision boundary plots require 2D features")
        
        # Create figure
        fig, ax = self.base_visualizer.create_figure(figsize=FIGURE_SIZES['decision_boundary'])
        
        # Create mesh for decision boundary
        x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
        y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                           np.arange(y_min, y_max, resolution))
        
        # Make predictions on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        predictions = model.predict(mesh_points)
        predictions = predictions.reshape(xx.shape)
        
        # Plot decision boundary
        colors = [EDUCATIONAL_COLORS['primary_blue'], EDUCATIONAL_COLORS['success_green']]
        ax.contourf(xx, yy, predictions, alpha=0.4, colors=colors, levels=1)
        
        # Plot data points
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(features[mask, 0], features[mask, 1], 
                      c=colors[i], label=f'Class {label}',
                      s=100, alpha=0.8, edgecolors='black', linewidth=2)
        
        # Apply styling
        title = title or f"{self.model_name} Decision Boundary"
        self.base_visualizer.apply_consistent_styling(
            ax=ax,
            title=title,
            xlabel="Feature 1",
            ylabel="Feature 2",
            grid=True
        )
        
        # Add educational annotation
        self._add_decision_boundary_annotation(ax)
        
        return fig, ax
    
    @performance_monitor
    def create_training_plot_optimized(self, 
                                     training_data: Dict[str, List[float]],
                                     plot_type: str = "learning_curve",
                                     title: Optional[str] = None,
                                     enable_caching: bool = True,
                                     **kwargs) -> Tuple[Figure.Figure, Axes]:
        """
        Create a training plot with performance optimizations.
        
        Args:
            training_data: Dictionary with training metrics
            plot_type: Type of training plot
            title: Optional custom title
            enable_caching: Whether to enable figure caching
            **kwargs: Additional plotting parameters
            
        Returns:
            Tuple of (figure, axes) with performance monitoring
        """
        # Check memory usage and cleanup if needed
        if self.memory_manager.should_cleanup():
            self.memory_manager.cleanup_memory()
        
        # Use lazy creation for large datasets
        data_size = sum(len(values) for values in training_data.values())
        if data_size > 1000:
            # Use lazy creation for large datasets
            return self._create_training_plot_lazy(training_data, plot_type, title, **kwargs)
        else:
            # Use standard creation for small datasets
            return self.create_training_plot(training_data, plot_type, title, **kwargs)
    
    def _create_training_plot_lazy(self, 
                                  training_data: Dict[str, List[float]],
                                  plot_type: str,
                                  title: Optional[str],
                                  **kwargs) -> Tuple[Figure.Figure, Axes]:
        """Create training plot using lazy creation for large datasets."""
        # Generate data hash for caching
        import hashlib
        data_str = str(training_data) + str(plot_type) + str(title)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()
        
        # Estimate data size
        data_size = sum(len(values) for values in training_data.values())
        
        def plot_func(**plot_kwargs):
            fig, ax = self.create_training_plot(training_data, plot_type, title, **plot_kwargs)
            return fig  # Only return the Figure, not the tuple
        
        fig = self.lazy_creator.create_plot_lazy(
            plot_func, data_hash, f"{plot_type}_training", data_size, **kwargs
        )
        
        # Get the axes from the figure
        ax = fig.gca()
        return fig, ax
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "plot_factory_performance": self.performance_monitor.get_performance_report(),
            "cache_stats": self.figure_cache.get_stats(),
            "memory_usage": self.memory_manager.check_memory_usage(),
            "optimization_recommendations": self._get_optimization_recommendations()
        }
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        # Check memory usage
        memory_info = self.memory_manager.check_memory_usage()
        if memory_info["usage_percent"] > 80:
            recommendations.append("High memory usage - consider enabling aggressive cleanup")
        
        # Check cache performance
        cache_stats = self.figure_cache.get_stats()
        if cache_stats["memory_usage_percent"] > 80:
            recommendations.append("Cache memory usage high - consider reducing cache size")
        
        # Check performance metrics
        perf_report = self.performance_monitor.get_performance_report()
        if "average_creation_time" in perf_report and perf_report["average_creation_time"] > 1.0:
            recommendations.append("Slow plot creation - consider using lazy plot creation")
        
        return recommendations
    
    def optimize_for_data_size(self, data_size: int) -> Dict[str, Any]:
        """Optimize settings for specific data size."""
        return self.memory_manager.optimize_for_data_size(data_size)
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """Clean up memory with optional aggressive cleanup."""
        return self.memory_manager.cleanup_memory(aggressive)
    
    def create_weight_evolution(self, 
                              weight_history: List[np.ndarray],
                              layer_names: Optional[List[str]] = None,
                              title: Optional[str] = None,
                              **kwargs) -> Tuple[Figure.Figure, List[Axes]]:
        """
        Create a standardized weight evolution visualization.
        
        Args:
            weight_history: List of weight arrays over time
            layer_names: Optional names for layers
            title: Optional custom title
            **kwargs: Additional plotting parameters
            
        Returns:
            Tuple of (figure, list of axes)
        """
        n_layers = len(weight_history[0]) if weight_history else 1
        
        # Create subplot layout
        if n_layers <= 4:
            fig, axes = self.base_visualizer.create_figure(
                figsize=FIGURE_SIZES['weight_visualization'],
                subplots=(2, 2)
            )
        else:
            cols = min(3, n_layers)
            rows = (n_layers + cols - 1) // cols
            fig, axes = self.base_visualizer.create_figure(
                figsize=(4*cols, 3*rows),
                subplots=(rows, cols)
            )
        
        # Flatten axes for easier iteration
        if n_layers == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot weight evolution for each layer
        epochs = np.arange(len(weight_history))
        for i in range(n_layers):
            ax = axes[i]
            
            # Extract weights for this layer
            layer_weights = [weights[i] for weights in weight_history]
            
            # Plot weight statistics
            mean_weights = [np.mean(w) for w in layer_weights]
            std_weights = [np.std(w) for w in layer_weights]
            
            ax.plot(epochs, mean_weights, 
                   label='Mean', color=EDUCATIONAL_COLORS['primary_blue'],
                   linewidth=2)
            ax.fill_between(epochs, 
                          [m - s for m, s in zip(mean_weights, std_weights)],
                          [m + s for m, s in zip(mean_weights, std_weights)],
                          alpha=0.3, color=EDUCATIONAL_COLORS['primary_blue'])
            
            # Apply styling
            layer_name = layer_names[i] if layer_names else f'Layer {i+1}'
            self.base_visualizer.apply_consistent_styling(
                ax=ax,
                title=f"{layer_name} Weight Evolution",
                xlabel="Epoch",
                ylabel="Weight Value",
                grid=True
            )
        
        # Hide unused subplots
        for i in range(n_layers, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        title = title or f"{self.model_name} Weight Evolution"
        fig.suptitle(title, fontsize=FONT_SIZES['title'], fontweight='bold')
        
        return fig, axes
    
    def create_confusion_matrix(self, 
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              title: Optional[str] = None,
                              **kwargs) -> Tuple[Figure.Figure, Axes]:
        """
        Create a standardized confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional names for classes
            title: Optional custom title
            **kwargs: Additional plotting parameters
            
        Returns:
            Tuple of (figure, axes)
        """
        from sklearn.metrics import confusion_matrix
        
        # Validate inputs
        self._validate_confusion_matrix_inputs(y_true, y_pred)
        
        # Create figure
        fig, ax = self.base_visualizer.create_figure(figsize=FIGURE_SIZES['confusion_matrix'])
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', alpha=0.8)
        
        # Add text annotations
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percentage = (count / total) * 100
                ax.text(j, i, f'{count}\n({percentage:.1f}%)',
                       ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black",
                       fontweight='bold', fontsize=10)
        
        # Apply styling
        title = title or f"{self.model_name} Confusion Matrix"
        self.base_visualizer.apply_consistent_styling(
            ax=ax,
            title=title,
            xlabel="Predicted Label",
            ylabel="True Label",
            grid=False
        )
        
        # Set tick labels
        if class_names:
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add educational annotation
        self._add_confusion_matrix_annotation(ax)
        
        return fig, ax
    
    def save_and_log(self, 
                    fig: Figure.Figure,
                    name: str,
                    plot_type: str,
                    model_info: Optional[Dict[str, Any]] = None,
                    dataset_info: Optional[Dict[str, Any]] = None,
                    hyperparameters: Optional[Dict[str, Any]] = None,
                    step: Optional[int] = None,
                    close_figure: bool = True) -> None:
        """
        Save figure and log to W&B with metadata.
        
        Args:
            fig: Figure to save and log
            name: Name for the figure
            plot_type: Type of plot for metadata
            model_info: Optional model information
            dataset_info: Optional dataset information
            hyperparameters: Optional hyperparameters
            step: Optional step number
            close_figure: Whether to close figure after logging
        """
        # Save locally
        local_path = self.save_dir / f"{name}.png"
        fig.savefig(local_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved: {local_path}")
        
        # Log to W&B if available
        if self.wandb_visualizer is not None:
            try:
                self.wandb_visualizer.log_figure_with_metadata(
                    figure=fig,
                    name=name,
                    plot_type=plot_type,
                    model_info=model_info,
                    dataset_info=dataset_info,
                    hyperparameters=hyperparameters,
                    step=step,
                    close_figure=close_figure
                )
            except Exception as e:
                logger.warning(f"Failed to log to W&B: {e}")
                if close_figure:
                    plt.close(fig)
        else:
            if close_figure:
                plt.close(fig)
    
    def _add_training_annotation(self, ax: Axes, plot_type: str) -> None:
        """Add educational annotation for training plots."""
        annotation_text = {
            'learning_curve': "Learning curves show how the model improves over time. Convergence indicates successful training.",
            'loss_curve': "Loss curves track prediction errors. Decreasing loss indicates learning progress.",
            'accuracy_curve': "Accuracy curves show classification performance. Higher accuracy indicates better predictions."
        }
        
        text = annotation_text.get(plot_type, "Training progress visualization.")
        self.base_visualizer.add_educational_annotation(
            ax=ax,
            text=text,
            position="top_right"
        )
    
    def _add_decision_boundary_annotation(self, ax: Axes) -> None:
        """Add educational annotation for decision boundary plots."""
        text = ("Decision boundaries show how the model separates classes. "
               "The colored regions indicate predicted class areas.")
        self.base_visualizer.add_educational_annotation(
            ax=ax,
            text=text,
            position="top_right"
        )
    
    def _add_confusion_matrix_annotation(self, ax: Axes) -> None:
        """Add educational annotation for confusion matrix plots."""
        text = ("Confusion matrices show prediction accuracy. "
               "Diagonal elements are correct predictions, off-diagonal are errors.")
        self.base_visualizer.add_educational_annotation(
            ax=ax,
            text=text,
            position="top_right"
        ) 