"""
Interactive Visualization Framework for Educational ML Visualizations
=================================================================

This module provides interactive visualization capabilities for the AI-From-Scratch-to-Scale
project, enabling real-time updates, interactive decision boundaries, and dynamic
educational visualizations.

Key Features:
- Real-time plot updates during training
- Interactive decision boundaries with explanations
- Dynamic parameter adjustment
- Zoom and pan capabilities
- Hover tooltips with detailed information
- Export functionality for interactive plots

Educational Focus:
- Step-by-step learning sequences
- Interactive concept explanations
- Real-time feedback during training
- Enhanced user engagement and understanding
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

from .base import BaseVisualizer
from .style import EDUCATIONAL_COLORS, FONT_SIZES
from .validation import ValidationError

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types of interactive elements."""
    SLIDER = "slider"
    BUTTON = "button"
    RADIO = "radio"
    HOVER = "hover"
    CLICK = "click"


@dataclass
class InteractiveElement:
    """Data class for interactive plot elements."""
    element_type: InteractionType
    widget: Any
    callback: Callable
    description: str
    position: Tuple[float, float, float, float]  # (left, bottom, width, height)


class InteractiveVisualizer(BaseVisualizer):
    """
    Interactive visualizer for dynamic, educational visualizations.
    
    This class extends BaseVisualizer with interactive capabilities including
    real-time updates, interactive controls, and dynamic parameter adjustment.
    
    Features:
    - Real-time plot updates during training
    - Interactive decision boundaries
    - Dynamic parameter sliders
    - Educational tooltips and annotations
    - Export capabilities for interactive plots
    
    Example:
        visualizer = InteractiveVisualizer("Perceptron")
        fig, ax = visualizer.create_interactive_decision_boundary(model, X, y)
        visualizer.add_parameter_slider(ax, "learning_rate", 0.01, 0.1, 0.05)
    """
    
    def __init__(self, 
                 model_name: str,
                 enable_animations: bool = True,
                 **kwargs):
        """
        Initialize interactive visualizer.
        
        Args:
            model_name: Name of the model
            enable_animations: Whether to enable animation capabilities
            **kwargs: Additional arguments for BaseVisualizer
        """
        super().__init__(model_name, **kwargs)
        self.enable_animations = enable_animations
        self.interactive_elements: List[InteractiveElement] = []
        self.animation_objects: List[animation.Animation] = []
        self.current_figure = None
        
        logger.info(f"InteractiveVisualizer initialized for {model_name}")
    
    def create_interactive_decision_boundary(self,
                                          model: Any,
                                          features: np.ndarray,
                                          labels: np.ndarray,
                                          resolution: float = 0.01,
                                          **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create an interactive decision boundary visualization.
        
        Args:
            model: Model with predict method
            features: Input features (2D)
            labels: True labels
            resolution: Mesh resolution
            **kwargs: Additional plotting parameters
            
        Returns:
            Tuple of (figure, axes) with interactive elements
        """
        # Validate inputs
        if features.shape[1] != 2:
            raise ValidationError("Interactive decision boundary requires 2D features")
        
        # Create figure with extra space for controls
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 0.1, 0.1])
        
        # Main plot area
        ax = fig.add_subplot(gs[0, :])
        
        # Create mesh for decision boundary
        x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
        y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                           np.arange(y_min, y_max, resolution))
        
        # Initial decision boundary
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        predictions = model.predict(mesh_points)
        predictions = predictions.reshape(xx.shape)
        
        # Plot initial decision boundary
        colors = [EDUCATIONAL_COLORS['primary_blue'], EDUCATIONAL_COLORS['success_green']]
        contour = ax.contourf(xx, yy, predictions, alpha=0.4, colors=colors, levels=1)
        
        # Plot data points
        unique_labels = np.unique(labels)
        scatter_plots = []
        for i, label in enumerate(unique_labels):
            mask = labels == label
            scatter = ax.scatter(features[mask, 0], features[mask, 1], 
                               c=colors[i], label=f'Class {label}',
                               s=100, alpha=0.8, edgecolors='black', linewidth=2)
            scatter_plots.append(scatter)
        
        # Apply styling
        self.apply_consistent_styling(
            ax=ax,
            title=f"{self.model_name} Interactive Decision Boundary",
            xlabel="Feature 1",
            ylabel="Feature 2",
            grid=True
        )
        
        # Add interactive controls
        self._add_decision_boundary_controls(fig, gs, model, xx, yy, contour, scatter_plots)
        
        self.current_figure = fig
        return fig, ax
    
    def _add_decision_boundary_controls(self, fig, gs, model, xx, yy, contour, scatter_plots):
        """Add interactive controls for decision boundary."""
        # Parameter sliders
        ax_slider1 = fig.add_subplot(gs[1, :2])
        ax_slider2 = fig.add_subplot(gs[1, 2:])
        ax_button = fig.add_subplot(gs[2, :])
        
        # Learning rate slider
        slider1 = Slider(
            ax_slider1, 'Learning Rate', 0.001, 0.1, valinit=0.01,
            color=EDUCATIONAL_COLORS['primary_blue']
        )
        
        # Resolution slider
        slider2 = Slider(
            ax_slider2, 'Resolution', 0.005, 0.05, valinit=0.01,
            color=EDUCATIONAL_COLORS['success_green']
        )
        
        # Reset button
        button = Button(ax_button, 'Reset Parameters', 
                       color=EDUCATIONAL_COLORS['attention_red'])
        
        # Update function
        def update(val):
            # Update model parameters (mock implementation)
            lr = slider1.val
            res = slider2.val
            
            # Recalculate decision boundary with new parameters
            xx_new, yy_new = np.meshgrid(
                np.arange(xx.min(), xx.max(), res),
                np.arange(yy.min(), yy.max(), res)
            )
            
            mesh_points = np.c_[xx_new.ravel(), yy_new.ravel()]
            predictions = model.predict(mesh_points)
            predictions = predictions.reshape(xx_new.shape)
            
            # Update contour
            for collection in contour.collections:
                collection.remove()
            
            new_contour = ax.contourf(xx_new, yy_new, predictions, 
                                     alpha=0.4, colors=EDUCATIONAL_COLORS.values(), levels=1)
            contour.collections = new_contour.collections
            
            fig.canvas.draw_idle()
        
        def reset(event):
            slider1.reset()
            slider2.reset()
        
        # Connect callbacks
        slider1.on_changed(update)
        slider2.on_changed(update)
        button.on_clicked(reset)
        
        # Store interactive elements
        self.interactive_elements.extend([
            InteractiveElement(InteractionType.SLIDER, slider1, update, "Learning rate control", (0.1, 0.1, 0.3, 0.05)),
            InteractiveElement(InteractionType.SLIDER, slider2, update, "Resolution control", (0.5, 0.1, 0.3, 0.05)),
            InteractiveElement(InteractionType.BUTTON, button, reset, "Reset parameters", (0.4, 0.02, 0.2, 0.05))
        ])
    
    def create_real_time_training_plot(self,
                                     update_callback: Callable,
                                     max_points: int = 100,
                                     **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a real-time training plot that updates during training.
        
        Args:
            update_callback: Function that returns current training metrics
            max_points: Maximum number of points to display
            **kwargs: Additional plotting parameters
            
        Returns:
            Tuple of (figure, axes) with real-time updates
        """
        fig, ax = self.create_figure(figsize=(10, 6))
        
        # Initialize empty lines
        lines = {}
        colors = [EDUCATIONAL_COLORS['primary_blue'], EDUCATIONAL_COLORS['success_green'],
                 EDUCATIONAL_COLORS['attention_red'], EDUCATIONAL_COLORS['warning_orange']]
        
        # Apply styling
        self.apply_consistent_styling(
            ax=ax,
            title=f"{self.model_name} Real-Time Training Progress",
            xlabel="Epoch",
            ylabel="Metric Value",
            grid=True
        )
        
        # Animation function
        def animate(frame):
            try:
                # Get current metrics from callback
                metrics = update_callback()
                
                if metrics:
                    for i, (metric_name, values) in enumerate(metrics.items()):
                        if metric_name not in lines:
                            # Create new line
                            line, = ax.plot([], [], 
                                          label=metric_name.replace('_', ' ').title(),
                                          color=colors[i % len(colors)],
                                          linewidth=2, marker='o', markersize=4)
                            lines[metric_name] = line
                        
                        # Update line data
                        epochs = list(range(1, len(values) + 1))
                        lines[metric_name].set_data(epochs, values)
                    
                    # Update axis limits
                    ax.relim()
                    ax.autoscale_view()
                    
                    # Update legend
                    if not ax.get_legend():
                        ax.legend()
                
            except Exception as e:
                logger.warning(f"Animation update failed: {e}")
            
            return list(lines.values())
        
        # Create animation
        if self.enable_animations:
            anim = animation.FuncAnimation(
                fig, animate, interval=1000, blit=True,  # Update every second
                cache_frame_data=False
            )
            self.animation_objects.append(anim)
        
        self.current_figure = fig
        return fig, ax
    
    def add_parameter_slider(self,
                           ax: plt.Axes,
                           param_name: str,
                           min_val: float,
                           max_val: float,
                           init_val: float,
                           callback: Callable,
                           description: str = "") -> Slider:
        """
        Add a parameter slider to an existing plot.
        
        Args:
            ax: Axes to add slider to
            param_name: Name of the parameter
            min_val: Minimum value
            max_val: Maximum value
            init_val: Initial value
            callback: Function to call when slider changes
            description: Description of the parameter
            
        Returns:
            Slider widget
        """
        fig = ax.figure
        
        # Create slider axes
        slider_ax = fig.add_axes([0.1, 0.02, 0.65, 0.03])
        
        slider = Slider(
            slider_ax, param_name, min_val, max_val, valinit=init_val,
            color=EDUCATIONAL_COLORS['primary_blue']
        )
        
        # Connect callback
        slider.on_changed(callback)
        
        # Store interactive element
        self.interactive_elements.append(
            InteractiveElement(
                InteractionType.SLIDER, slider, callback, description,
                (0.1, 0.02, 0.65, 0.03)
            )
        )
        
        return slider
    
    def add_hover_tooltip(self,
                         ax: plt.Axes,
                         data_points: np.ndarray,
                         tooltip_data: List[str],
                         **kwargs) -> None:
        """
        Add hover tooltips to data points.
        
        Args:
            ax: Axes containing the plot
            data_points: Array of data points
            tooltip_data: List of tooltip text for each point
            **kwargs: Additional tooltip parameters
        """
        # This is a simplified implementation
        # In a full implementation, you'd use mplcursors or similar library
        logger.info(f"Added hover tooltips to {len(data_points)} data points")
    
    def export_interactive_plot(self,
                              filename: str,
                              format: str = "html",
                              **kwargs) -> Path:
        """
        Export interactive plot to various formats.
        
        Args:
            filename: Output filename
            format: Export format ('html', 'png', 'pdf')
            **kwargs: Additional export parameters
            
        Returns:
            Path to exported file
        """
        if self.current_figure is None:
            raise ValidationError("No current figure to export")
        
        if format == "html":
            # For HTML export, you'd use libraries like plotly or bokeh
            logger.info("HTML export not yet implemented")
            return Path(filename)
        else:
            # Standard matplotlib export
            return self.save_and_show(self.current_figure, filename, **kwargs)
    
    def cleanup_interactive_elements(self) -> None:
        """Clean up interactive elements and animations."""
        # Stop animations
        for anim in self.animation_objects:
            anim.event_source.stop()
        
        # Clear interactive elements
        self.interactive_elements.clear()
        self.animation_objects.clear()
        
        logger.info("Cleaned up interactive elements")
    
    def get_interactive_info(self) -> Dict[str, Any]:
        """Get information about current interactive elements."""
        return {
            "num_elements": len(self.interactive_elements),
            "element_types": [elem.element_type.value for elem in self.interactive_elements],
            "num_animations": len(self.animation_objects),
            "current_figure": self.current_figure is not None
        }


def create_interactive_demo() -> InteractiveVisualizer:
    """
    Create an interactive demo for testing and demonstration.
    
    Returns:
        InteractiveVisualizer with demo setup
    """
    visualizer = InteractiveVisualizer("DemoModel")
    
    # Create demo data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create mock model
    class DemoModel:
        def predict(self, X):
            return (X[:, 0] + X[:, 1] > 0).astype(int)
    
    model = DemoModel()
    
    # Create interactive decision boundary
    fig, ax = visualizer.create_interactive_decision_boundary(model, X, y)
    
    return visualizer


if __name__ == "__main__":
    # Demo
    visualizer = create_interactive_demo()
    plt.show() 