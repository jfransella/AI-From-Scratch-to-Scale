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
        
        logger.debug(f"Initialized {model_name}Visualizer with {style_theme} theme")
    
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
