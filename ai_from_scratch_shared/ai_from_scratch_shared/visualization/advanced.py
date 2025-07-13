"""
Advanced Visualization Types for Educational ML Visualizations
============================================================

This module provides advanced visualization capabilities for the AI-From-Scratch-to-Scale
project, including gradient flow, attention mechanisms, feature importance, and
other sophisticated plot types for deep learning education.

Key Features:
- Gradient flow visualization for neural networks
- Attention mechanism heatmaps for transformers
- Feature importance and interpretability plots
- Learning rate scheduling visualization
- Model comparison dashboards
- 3D visualizations for multi-dimensional data

Educational Focus:
- Deep learning concept visualization
- Model interpretability and explainability
- Advanced neural network architectures
- Interactive learning experiences
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

from .base import BaseVisualizer
from .style import EDUCATIONAL_COLORS, FONT_SIZES, FIGURE_SIZES
from .validation import ValidationError

logger = logging.getLogger(__name__)


class PlotType(Enum):
    """Types of advanced plots."""
    GRADIENT_FLOW = "gradient_flow"
    ATTENTION_HEATMAP = "attention_heatmap"
    FEATURE_IMPORTANCE = "feature_importance"
    LEARNING_RATE_SCHEDULE = "learning_rate_schedule"
    MODEL_COMPARISON = "model_comparison"
    NETWORK_ARCHITECTURE = "network_architecture"
    LOSS_LANDSCAPE = "loss_landscape"


@dataclass
class LayerInfo:
    """Information about a neural network layer."""
    name: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    activation: str
    parameters: int


class AdvancedVisualizer(BaseVisualizer):
    """
    Advanced visualizer for sophisticated ML visualizations.
    
    This class provides advanced visualization capabilities including
    gradient flow, attention mechanisms, feature importance, and
    other deep learning specific visualizations.
    
    Features:
    - Gradient flow visualization for neural networks
    - Attention mechanism heatmaps
    - Feature importance plots
    - Learning rate scheduling visualization
    - Model comparison dashboards
    - 3D visualizations
    
    Example:
        visualizer = AdvancedVisualizer("Transformer")
        fig, ax = visualizer.create_attention_heatmap(attention_weights)
        fig, ax = visualizer.create_gradient_flow(gradients, layer_names)
    """
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize advanced visualizer."""
        super().__init__(model_name, **kwargs)
        logger.info(f"AdvancedVisualizer initialized for {model_name}")
    
    def create_gradient_flow(self,
                           gradients: List[np.ndarray],
                           layer_names: Optional[List[str]] = None,
                           title: Optional[str] = None,
                           **kwargs) -> Tuple[Figure, Axes]:
        """
        Create gradient flow visualization for neural networks.
        
        Args:
            gradients: List of gradient arrays for each layer
            layer_names: Names of layers (optional)
            title: Custom title
            **kwargs: Additional plotting parameters
            
        Returns:
            Tuple of (figure, axes)
        """
        if not gradients:
            raise ValidationError("Gradients list cannot be empty")
        
        # Create figure
        fig, ax = self.create_figure(figsize=FIGURE_SIZES.get('gradient_flow', (12, 8)))
        
        # Prepare data
        n_layers = len(gradients)
        layer_names = layer_names or [f"Layer {i+1}" for i in range(n_layers)]
        
        # Calculate gradient statistics
        gradient_norms = [np.linalg.norm(grad) for grad in gradients]
        gradient_means = [np.mean(grad) for grad in gradients]
        gradient_stds = [np.std(grad) for grad in gradients]
        
        # Create gradient flow plot
        x_pos = np.arange(n_layers)
        bars = ax.bar(x_pos, gradient_norms, 
                     color=EDUCATIONAL_COLORS['primary_blue'],
                     alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add error bars for gradient statistics
        ax.errorbar(x_pos, gradient_means, yerr=gradient_stds,
                   fmt='o', color=EDUCATIONAL_COLORS['attention_red'],
                   capsize=5, capthick=2, markersize=8)
        
        # Add gradient flow arrows
        for i in range(n_layers - 1):
            ax.annotate('', xy=(i + 1, gradient_norms[i + 1]),
                       xytext=(i, gradient_norms[i]),
                       arrowprops=dict(arrowstyle='->', 
                                     color=EDUCATIONAL_COLORS['success_green'],
                                     lw=2, alpha=0.8))
        
        # Apply styling
        title = title or f"{self.model_name} Gradient Flow"
        self.apply_consistent_styling(
            ax=ax,
            title=title,
            xlabel="Layer",
            ylabel="Gradient Norm",
            grid=True
        )
        
        # Set x-axis labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        
        # Add educational annotation
        self.add_educational_annotation(
            ax=ax,
            text="Gradient flow shows how gradients propagate through the network. "
                 "Vanishing gradients appear as decreasing norms.",
            position="top_right"
        )
        
        return fig, ax
    
    def create_attention_heatmap(self,
                               attention_weights: np.ndarray,
                               input_tokens: Optional[List[str]] = None,
                               output_tokens: Optional[List[str]] = None,
                               title: Optional[str] = None,
                               **kwargs) -> Tuple[Figure, Axes]:
        """
        Create attention mechanism heatmap for transformers.
        
        Args:
            attention_weights: Attention weight matrix (n_tokens x n_tokens)
            input_tokens: Input token names (optional)
            output_tokens: Output token names (optional)
            title: Custom title
            **kwargs: Additional plotting parameters
            
        Returns:
            Tuple of (figure, axes)
        """
        if attention_weights.ndim != 2:
            raise ValidationError("Attention weights must be 2D matrix")
        
        if attention_weights.shape[0] != attention_weights.shape[1]:
            raise ValidationError("Attention weights must be square matrix")
        
        # Create figure
        fig, ax = self.create_figure(figsize=FIGURE_SIZES.get('attention_heatmap', (10, 8)))
        
        # Create heatmap
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto', alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)
        
        # Set tick labels
        n_tokens = attention_weights.shape[0]
        if input_tokens is None:
            input_tokens = [f"Token {i}" for i in range(n_tokens)]
        if output_tokens is None:
            output_tokens = [f"Token {i}" for i in range(n_tokens)]
        
        ax.set_xticks(range(n_tokens))
        ax.set_yticks(range(n_tokens))
        ax.set_xticklabels(output_tokens, rotation=45, ha='right')
        ax.set_yticklabels(input_tokens)
        
        # Add text annotations for high attention weights
        threshold = np.max(attention_weights) * 0.5
        for i in range(n_tokens):
            for j in range(n_tokens):
                weight = attention_weights[i, j]
                if weight > threshold:
                    ax.text(j, i, f'{weight:.2f}',
                           ha="center", va="center",
                           color="white" if weight > threshold * 1.5 else "black",
                           fontweight='bold', fontsize=8)
        
        # Apply styling
        title = title or f"{self.model_name} Attention Mechanism"
        self.apply_consistent_styling(
            ax=ax,
            title=title,
            xlabel="Output Tokens",
            ylabel="Input Tokens",
            grid=False
        )
        
        # Add educational annotation
        self.add_educational_annotation(
            ax=ax,
            text="Attention heatmaps show which input tokens the model focuses on "
                 "when generating each output token. Brighter colors indicate higher attention.",
            position="top_right"
        )
        
        return fig, ax
    
    def create_feature_importance(self,
                                feature_names: List[str],
                                importance_scores: np.ndarray,
                                method: str = "bar",
                                title: Optional[str] = None,
                                **kwargs) -> Tuple[Figure, Axes]:
        """
        Create feature importance visualization.
        
        Args:
            feature_names: Names of features
            importance_scores: Importance scores for each feature
            method: Visualization method ('bar', 'horizontal', 'radar')
            title: Custom title
            **kwargs: Additional plotting parameters
            
        Returns:
            Tuple of (figure, axes)
        """
        if len(feature_names) != len(importance_scores):
            raise ValidationError("Feature names and importance scores must have same length")
        
        # Create figure
        fig, ax = self.create_figure(figsize=FIGURE_SIZES.get('feature_importance', (10, 8)))
        
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_scores = importance_scores[sorted_indices]
        
        if method == "bar":
            # Vertical bar chart
            bars = ax.bar(range(len(sorted_names)), sorted_scores,
                         color=EDUCATIONAL_COLORS['primary_blue'],
                         alpha=0.7, edgecolor='black', linewidth=1)
            
            ax.set_xticks(range(len(sorted_names)))
            ax.set_xticklabels(sorted_names, rotation=45, ha='right')
            ax.set_ylabel("Importance Score")
            
        elif method == "horizontal":
            # Horizontal bar chart
            y_pos = np.arange(len(sorted_names))
            bars = ax.barh(y_pos, sorted_scores,
                          color=EDUCATIONAL_COLORS['success_green'],
                          alpha=0.7, edgecolor='black', linewidth=1)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names)
            ax.set_xlabel("Importance Score")
            
        else:
            raise ValidationError(f"Unknown method: {method}")
        
        # Apply styling
        title = title or f"{self.model_name} Feature Importance"
        self.apply_consistent_styling(
            ax=ax,
            title=title,
            xlabel="Feature" if method == "bar" else "Importance Score",
            ylabel="Importance Score" if method == "bar" else "Feature",
            grid=True
        )
        
        # Add educational annotation
        self.add_educational_annotation(
            ax=ax,
            text="Feature importance shows which input features have the most "
                 "influence on the model's predictions.",
            position="top_right"
        )
        
        return fig, ax
    
    def create_learning_rate_schedule(self,
                                    epochs: List[int],
                                    learning_rates: List[float],
                                    schedule_type: str = "step",
                                    title: Optional[str] = None,
                                    **kwargs) -> Tuple[Figure, Axes]:
        """
        Create learning rate scheduling visualization.
        
        Args:
            epochs: List of epoch numbers
            learning_rates: Learning rates at each epoch
            schedule_type: Type of schedule ('step', 'exponential', 'cosine')
            title: Custom title
            **kwargs: Additional plotting parameters
            
        Returns:
            Tuple of (figure, axes)
        """
        if len(epochs) != len(learning_rates):
            raise ValidationError("Epochs and learning rates must have same length")
        
        # Create figure
        fig, ax = self.create_figure(figsize=FIGURE_SIZES.get('learning_rate_schedule', (10, 6)))
        
        # Plot learning rate schedule
        ax.plot(epochs, learning_rates, 
               color=EDUCATIONAL_COLORS['primary_blue'],
               linewidth=3, marker='o', markersize=6, alpha=0.8)
        
        # Add schedule type annotation
        ax.text(0.02, 0.98, f"Schedule: {schedule_type.title()}",
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=EDUCATIONAL_COLORS['neutral_light'],
                        edgecolor=EDUCATIONAL_COLORS['primary_blue']))
        
        # Apply styling
        title = title or f"{self.model_name} Learning Rate Schedule"
        self.apply_consistent_styling(
            ax=ax,
            title=title,
            xlabel="Epoch",
            ylabel="Learning Rate",
            grid=True
        )
        
        # Use log scale for learning rate if appropriate
        if max(learning_rates) / min(learning_rates) > 10:
            ax.set_yscale('log')
        
        # Add educational annotation
        self.add_educational_annotation(
            ax=ax,
            text="Learning rate scheduling helps optimize training by adjusting "
                 "the learning rate over time for better convergence.",
            position="top_right"
        )
        
        return fig, ax
    
    def create_model_comparison(self,
                              model_names: List[str],
                              metrics: Dict[str, List[float]],
                              comparison_type: str = "bar",
                              title: Optional[str] = None,
                              **kwargs) -> Tuple[Figure, Union[Axes, List[Axes]]]:
        """
        Create model comparison visualization.
        
        Args:
            model_names: Names of models to compare
            metrics: Dictionary of metric names to lists of values
            comparison_type: Type of comparison ('bar', 'radar', 'table')
            title: Custom title
            **kwargs: Additional plotting parameters
            
        Returns:
            Tuple of (figure, axes)
        """
        if not model_names or not metrics:
            raise ValidationError("Model names and metrics cannot be empty")
        
        # Create figure
        if comparison_type == "radar":
            fig, axes = self.create_figure(figsize=(12, 8), subplots=(1, len(metrics)))
            if len(metrics) == 1:
                axes = [axes]
        else:
            fig, ax = self.create_figure(figsize=FIGURE_SIZES.get('model_comparison', (12, 8)))
        
        if comparison_type == "bar":
            # Bar chart comparison
            x = np.arange(len(model_names))
            width = 0.8 / len(metrics)
            
            colors = [EDUCATIONAL_COLORS['primary_blue'], EDUCATIONAL_COLORS['success_green'],
                     EDUCATIONAL_COLORS['attention_red'], EDUCATIONAL_COLORS['warning_orange']]
            
            for i, (metric_name, values) in enumerate(metrics.items()):
                if len(values) != len(model_names):
                    raise ValidationError(f"Metric {metric_name} must have same length as model_names")
                
                ax.bar(x + i * width, values, width,
                      label=metric_name.replace('_', ' ').title(),
                      color=colors[i % len(colors)], alpha=0.7)
            
            ax.set_xticks(x + width * (len(metrics) - 1) / 2)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.legend()
            ax.set_ylabel("Metric Value")
            
        elif comparison_type == "radar":
            # Radar chart comparison (simplified)
            for i, (metric_name, values) in enumerate(metrics.items()):
                ax = axes[i] if len(metrics) > 1 else axes
                
                # Create simple radar-like visualization
                angles = np.linspace(0, 2 * np.pi, len(model_names), endpoint=False)
                values_normalized = np.array(values) / max(values)
                
                ax.plot(angles, values_normalized, 'o-', linewidth=2, markersize=8)
                ax.set_xticks(angles)
                ax.set_xticklabels(model_names)
                ax.set_ylim(0, 1)
                ax.set_title(metric_name.replace('_', ' ').title())
                ax.grid(True)
        
        else:
            raise ValidationError(f"Unknown comparison type: {comparison_type}")
        
        # Apply styling
        title = title or f"{self.model_name} Model Comparison"
        if comparison_type == "bar":
            self.apply_consistent_styling(
                ax=ax,
                title=title,
                xlabel="Model",
                ylabel="Metric Value",
                grid=True
            )
        
        # Add educational annotation
        if comparison_type == "bar":
            self.add_educational_annotation(
                ax=ax,
                text="Model comparison helps evaluate different architectures "
                     "and hyperparameter settings for optimal performance.",
                position="top_right"
            )
        
        return fig, (axes if comparison_type == "radar" else ax)
    
    def create_network_architecture(self,
                                  layers: List[LayerInfo],
                                  title: Optional[str] = None,
                                  **kwargs) -> Tuple[Figure, Axes]:
        """
        Create neural network architecture visualization.
        
        Args:
            layers: List of layer information
            title: Custom title
            **kwargs: Additional plotting parameters
            
        Returns:
            Tuple of (figure, axes)
        """
        if not layers:
            raise ValidationError("Layers list cannot be empty")
        
        # Create figure
        fig, ax = self.create_figure(figsize=FIGURE_SIZES.get('network_architecture', (12, 8)))
        
        # Calculate layout
        n_layers = len(layers)
        layer_width = 0.8 / n_layers
        
        # Draw layers
        for i, layer in enumerate(layers):
            x = 0.1 + i * layer_width
            y = 0.1
            width = layer_width * 0.8
            height = 0.8
            
            # Create layer rectangle
            rect = patches.Rectangle((x, y), width, height,
                                   linewidth=2, edgecolor=EDUCATIONAL_COLORS['primary_blue'],
                                   facecolor=EDUCATIONAL_COLORS['neutral_light'], alpha=0.7)
            ax.add_patch(rect)
            
            # Add layer name
            ax.text(x + width/2, y + height + 0.02, layer.name,
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Add layer info
            info_text = f"Input: {layer.input_shape}\nOutput: {layer.output_shape}\nActivation: {layer.activation}\nParams: {layer.parameters:,}"
            ax.text(x + width/2, y + height/2, info_text,
                   ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Draw connections to next layer
            if i < n_layers - 1:
                next_x = x + width
                next_y = y + height/2
                ax.annotate('', xy=(next_x + layer_width * 0.8, next_y),
                           xytext=(next_x, next_y),
                           arrowprops=dict(arrowstyle='->', 
                                         color=EDUCATIONAL_COLORS['success_green'],
                                         lw=2, alpha=0.8))
        
        # Apply styling
        title = title or f"{self.model_name} Network Architecture"
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=FONT_SIZES['title'], fontweight='bold')
        ax.axis('off')
        
        # Add educational annotation
        self.add_educational_annotation(
            ax=ax,
            text="Network architecture shows the structure and connectivity "
                 "of neural network layers.",
            position="top_right"
        )
        
        return fig, ax


def create_advanced_demo() -> AdvancedVisualizer:
    """
    Create an advanced demo for testing and demonstration.
    
    Returns:
        AdvancedVisualizer with demo setup
    """
    visualizer = AdvancedVisualizer("DemoModel")
    
    # Create demo data
    np.random.seed(42)
    
    # Gradient flow demo
    gradients = [np.random.randn(10, 10) for _ in range(5)]
    layer_names = ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Output"]
    
    # Feature importance demo
    feature_names = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"]
    importance_scores = np.random.rand(5)
    
    # Learning rate schedule demo
    epochs = list(range(1, 21))
    learning_rates = [0.1 * (0.9 ** i) for i in range(20)]
    
    # Create visualizations
    fig1, ax1 = visualizer.create_gradient_flow(gradients, layer_names)
    fig2, ax2 = visualizer.create_feature_importance(feature_names, importance_scores)
    fig3, ax3 = visualizer.create_learning_rate_schedule(epochs, learning_rates)
    
    return visualizer


if __name__ == "__main__":
    # Demo
    visualizer = create_advanced_demo()
    plt.show() 