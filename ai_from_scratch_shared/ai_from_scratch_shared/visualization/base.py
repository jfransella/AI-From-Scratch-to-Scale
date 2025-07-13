"""
Base Visualizer Class for Educational ML Visualizations
======================================================

This module provides the foundational BaseVisualizer class that all model-specific
visualizers inherit from. It ensures consistent behavior, styling, and educational
value across all visualizations in the AI-From-Scratch-to-Scale project.

Key Features:
- Consistent figure creation and management
- Educational annotation framework
- Unified save/show logic with proper path handling
- Model-specific color scheme integration
- Error handling and validation
- Integration with W&B logging systems

Educational Philosophy:
- Every visualization should teach something specific
- Consistent styling reduces cognitive load for learners
- Annotations provide mathematical and conceptual context
- Modular design allows progressive complexity
"""

import matplotlib.pyplot as plt
import matplotlib.figure as Figure
from matplotlib.axes import Axes
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
import logging
import warnings
import numpy as np

from .style import (
    FIGURE_SIZES,
    get_model_color_scheme,
    setup_educational_style,
    EDUCATIONAL_COLORS
)
from .utils import save_and_show_plot, format_axes_for_education
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


class BaseVisualizer:
    """
    Base class for all model-specific visualizers in the project.
    
    This class provides common functionality for creating educational
    visualizations with consistent styling and behavior.
    
    Features:
    - Automatic figure creation with educational styling
    - Model-specific color schemes
    - Educational annotation support
    - Consistent save/show behavior
    - Integration with experiment tracking
    
    Example:
        class PerceptronVisualizer(BaseVisualizer):
            def __init__(self):
                super().__init__(model_name="Perceptron")
            
            def plot_decision_boundary(self, model, X, y):
                fig, ax = self.create_figure(figsize='decision_boundary')
                # Implementation here...
                return self.save_and_show(fig, "decision_boundary.png")
    """
    
    def __init__(self, 
                 model_name: str,
                 style_theme: str = "educational",
                 default_save_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the base visualizer.
        
        Args:
            model_name: Name of the model (e.g., "Perceptron", "MLP", "Hopfield")
            style_theme: Theme to use ("educational" or "professional")
            default_save_dir: Default directory for saving plots
        """
        self.model_name = model_name
        self.style_theme = style_theme
        self.default_save_dir = Path(default_save_dir) if default_save_dir else None
        
        # Get model-specific color scheme
        self.colors = get_model_color_scheme(model_name)
        
        # Initialize styling
        setup_educational_style()
        
        # Track created figures for cleanup
        self._figures = []
        
        # Initialize performance optimization components
        self.performance_monitor = PerformanceMonitor()
        self.figure_cache = FigureCache()
        self.lazy_creator = LazyPlotCreator(self.figure_cache)
        self.memory_manager = MemoryManager()
        
        logger.debug(f"Initialized {model_name}Visualizer with {style_theme} theme")
    
    def validate_inputs(self, **kwargs) -> None:
        """
        Validate common input parameters with detailed error messages.
        
        Args:
            **kwargs: Input parameters to validate
            
        Raises:
            ValidationError: If inputs are invalid
        """
        validator = VisualizationValidator()
        
        # Validate model_name
        if 'model_name' in kwargs:
            model_name = kwargs['model_name']
            if not isinstance(model_name, str):
                raise ValidationError(
                    f"model_name must be a string, got {type(model_name)}",
                    [
                        "Use descriptive string for model name",
                        "Example: model_name='Perceptron'"
                    ]
                )
            if len(model_name.strip()) == 0:
                raise ValidationError(
                    "model_name cannot be empty",
                    [
                        "Provide meaningful model name",
                        "Use descriptive name like 'Perceptron' or 'MLP'"
                    ]
                )
        
        # Validate style_theme
        if 'style_theme' in kwargs:
            theme = kwargs['style_theme']
            valid_themes = ['educational', 'professional']
            if theme not in valid_themes:
                raise ValidationError(
                    f"style_theme must be one of {valid_themes}, got '{theme}'",
                    [
                        f"Use one of: {', '.join(valid_themes)}",
                        "Default is 'educational' for learning-focused plots"
                    ]
                )
    
    def validate_data_for_visualization(self, 
                                      features: np.ndarray,
                                      labels: np.ndarray,
                                      model: Any = None) -> None:
        """
        Validate data inputs for visualization methods.
        
        Args:
            features: Input features
            labels: Target labels
            model: Model object (optional)
            
        Raises:
            ValidationError: If data is invalid for visualization
        """
        validator = VisualizationValidator()
        
        # Validate features
        if features is not None:
            validator.validate_2d_features(features, "features")
        
        # Validate labels
        if labels is not None:
            validator.validate_labels(labels, data_name="labels")
        
        # Validate model if provided
        if model is not None:
            validator.validate_model_interface(model)
        
        # Validate matching lengths
        if features is not None and labels is not None:
            if len(features) != len(labels):
                raise ValidationError(
                    f"Features and labels must have same length: {len(features)} vs {len(labels)}",
                    [
                        "Check data loading and preprocessing",
                        "Ensure features and labels are aligned",
                        "Verify data splitting and shuffling"
                    ]
                )
    
    def create_figure(self, 
                     figsize: Union[Tuple[int, int], str] = 'default',
                     subplots: Tuple[int, int] = (1, 1),
                     **kwargs) -> Tuple[Figure.Figure, Union[Axes, np.ndarray]]:
        """
        Create a figure with consistent educational styling.
        
        Args:
            figsize: Figure size as tuple or preset name from FIGURE_SIZES
            subplots: Number of subplots as (rows, cols)
            **kwargs: Additional arguments passed to plt.subplots()
            
        Returns:
            Tuple of (figure, axes) - axes is single Axes if (1,1), array otherwise
            
        Example:
            # Create single plot
            fig, ax = self.create_figure(figsize='confusion_matrix')
            
            # Create subplot grid
            fig, axes = self.create_figure(figsize=(12, 8), subplots=(2, 2))
        """
        # Resolve figsize
        if isinstance(figsize, str):
            if figsize in FIGURE_SIZES:
                figsize = FIGURE_SIZES[figsize]
            else:
                logger.warning(f"Unknown figsize preset '{figsize}', using default")
                figsize = FIGURE_SIZES['default']
        
        # Create figure with subplots
        fig, axes = plt.subplots(
            subplots[0], subplots[1],
            figsize=figsize,
            facecolor='white',
            **kwargs
        )
        
        # Apply educational formatting to all axes
        if subplots == (1, 1):
            format_axes_for_education(axes, self.colors)
        else:
            # Handle multiple subplots
            if hasattr(axes, 'flat'):
                for ax in axes.flat:
                    format_axes_for_education(ax, self.colors)
            else:
                # Single row or column
                for ax in axes:
                    format_axes_for_education(ax, self.colors)
        
        # Track figure for cleanup
        self._figures.append(fig)
        
        return fig, axes
    
    def save_and_show(self, 
                     fig: Figure.Figure,
                     filename: Optional[str] = None,
                     save_path: Optional[Union[str, Path]] = None,
                     show: bool = True,
                     close_after: bool = False,
                     **kwargs) -> Optional[Path]:
        """
        Save and optionally show a figure with consistent behavior.
        
        Args:
            fig: Figure to save/show
            filename: Name of file to save (if saving)
            save_path: Full path to save file (overrides filename)
            show: Whether to display the figure
            close_after: Whether to close figure after saving/showing
            **kwargs: Additional arguments for plt.savefig()
            
        Returns:
            Path where figure was saved (if saved), None otherwise
            
        Example:
            fig, ax = self.create_figure()
            ax.plot([1, 2, 3], [1, 4, 2])
            saved_path = self.save_and_show(fig, "my_plot.png", show=True)
        """
        return save_and_show_plot(
            fig=fig,
            filename=filename,
            save_path=save_path,
            default_dir=self.default_save_dir,
            show=show,
            close_after=close_after,
            **kwargs
        )
    
    @performance_monitor
    def create_figure_optimized(self, 
                               figsize: Union[Tuple[int, int], str] = 'default',
                               subplots: Tuple[int, int] = (1, 1),
                               enable_caching: bool = True,
                               **kwargs) -> Tuple[Figure.Figure, Union[Axes, np.ndarray]]:
        """
        Create a figure with performance optimizations.
        
        Args:
            figsize: Figure size as tuple or preset name
            subplots: Number of subplots as (rows, cols)
            enable_caching: Whether to enable figure caching
            **kwargs: Additional arguments passed to plt.subplots()
            
        Returns:
            Tuple of (figure, axes) with performance monitoring
        """
        # Check memory usage and cleanup if needed
        if self.memory_manager.should_cleanup():
            self.memory_manager.cleanup_memory()
        
        # Create figure with standard method
        fig, axes = self.create_figure(figsize, subplots, **kwargs)
        
        # Cache the figure if enabled
        if enable_caching:
            # Generate cache key based on parameters
            cache_key = f"{figsize}_{subplots}_{hash(str(kwargs))}"
            self.figure_cache.put("figure", cache_key, fig)
        
        return fig, axes
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "visualizer_performance": self.performance_monitor.get_performance_report(),
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
    
    def add_educational_annotation(self, 
                                  ax: Axes,
                                  text: str,
                                  position: str = "top_right",
                                  box_style: str = "round,pad=0.3",
                                  **kwargs) -> None:
        """
        Add educational annotation to a plot.
        
        Args:
            ax: Axes to annotate
            text: Annotation text
            position: Position preset ("top_right", "top_left", "bottom_right", "bottom_left")
            box_style: Style for annotation box
            **kwargs: Additional arguments for ax.annotate()
        """
        # Define position coordinates
        positions = {
            "top_right": (0.95, 0.95),
            "top_left": (0.05, 0.95),
            "bottom_right": (0.95, 0.05),
            "bottom_left": (0.05, 0.05),
        }
        
        if position not in positions:
            logger.warning(f"Unknown position '{position}', using 'top_right'")
            position = "top_right"
        
        x, y = positions[position]
        
        # Default annotation styling
        annotation_kwargs = {
            'xy': (x, y),
            'xycoords': 'axes fraction',
            'bbox': {
                'boxstyle': box_style,
                'facecolor': self.colors['background'],
                'edgecolor': self.colors['primary'],
                'alpha': 0.8
            },
            'fontsize': 9,
            'ha': 'right' if 'right' in position else 'left',
            'va': 'top' if 'top' in position else 'bottom',
            'color': self.colors['text']
        }
        
        # Override with user-provided kwargs
        annotation_kwargs.update(kwargs)
        
        ax.annotate(text, **annotation_kwargs)
    
    def apply_consistent_styling(self,
                               ax: Axes,
                               title: str,
                               xlabel: str,
                               ylabel: str,
                               grid: bool = True) -> None:
        """
        Apply consistent styling to an axes object.
        
        Args:
            ax: Axes to style
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            grid: Whether to show grid
        """
        # Set labels and title
        ax.set_title(title, color=self.colors['text'], fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, color=self.colors['text'])
        ax.set_ylabel(ylabel, color=self.colors['text'])
        
        # Grid styling
        if grid:
            ax.grid(True, alpha=0.3, color=self.colors['text'])
        
        # Spine styling
        for spine in ax.spines.values():
            spine.set_color(self.colors['text'])
            spine.set_linewidth(1.0)
        
        # Tick styling
        ax.tick_params(colors=self.colors['text'])
    
    def create_comparison_figure(self,
                               n_comparisons: int,
                               layout: str = "auto") -> Tuple[Figure.Figure, List[Axes]]:
        """
        Create figure optimized for comparing multiple visualizations.
        
        Args:
            n_comparisons: Number of comparisons to show
            layout: Layout style ("horizontal", "vertical", "grid", "auto")
            
        Returns:
            Tuple of (figure, list of axes)
        """
        if layout == "auto":
            if n_comparisons <= 2:
                layout = "horizontal"
            elif n_comparisons <= 4:
                layout = "grid"
            else:
                layout = "grid"
        
        if layout == "horizontal":
            rows, cols = 1, n_comparisons
            figsize = (4 * n_comparisons, 6)
        elif layout == "vertical":
            rows, cols = n_comparisons, 1
            figsize = (8, 4 * n_comparisons)
        elif layout == "grid":
            cols = int(np.ceil(np.sqrt(n_comparisons)))
            rows = int(np.ceil(n_comparisons / cols))
            figsize = (4 * cols, 4 * rows)
        else:
            raise ValueError(f"Unknown layout: {layout}")
        
        fig, axes = self.create_figure(figsize=figsize, subplots=(rows, cols))
        
        # Ensure axes is always a list
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten()
        
        # Hide unused subplots
        if len(axes) > n_comparisons:
            for i in range(n_comparisons, len(axes)):
                axes[i].set_visible(False)
        
        return fig, axes[:n_comparisons]
    
    def add_model_watermark(self, fig: Figure.Figure) -> None:
        """
        Add subtle model identification watermark to figure.
        
        Args:
            fig: Figure to add watermark to
        """
        fig.text(
            0.99, 0.01,
            f"AI-From-Scratch-to-Scale: {self.model_name}",
            ha='right', va='bottom',
            fontsize=8,
            alpha=0.5,
            color=self.colors['text']
        )
    
    def cleanup_figures(self) -> None:
        """Close all figures created by this visualizer to free memory."""
        for fig in self._figures:
            plt.close(fig)
        self._figures.clear()
        logger.debug(f"Cleaned up {len(self._figures)} figures")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup figures."""
        self.cleanup_figures()
    
    def get_figure_info(self) -> Dict[str, Any]:
        """
        Get information about the visualizer configuration.
        
        Returns:
            Dictionary with visualizer configuration
        """
        return {
            'model_name': self.model_name,
            'style_theme': self.style_theme,
            'default_save_dir': str(self.default_save_dir) if self.default_save_dir else None,
            'color_scheme': self.colors,
            'active_figures': len(self._figures)
        }
    
    def __repr__(self) -> str:
        """String representation of the visualizer."""
        return f"{self.__class__.__name__}(model='{self.model_name}', theme='{self.style_theme}')"
