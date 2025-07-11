"""
Educational Annotation Framework for ML Visualizations
=====================================================

This module provides functions and classes for adding educational context
to machine learning visualizations. The goal is to help learners understand
the mathematical and conceptual foundations behind each visualization.

Key Features:
- Mathematical context annotations
- Performance insights and interpretations
- Concept explanations and learning objectives
- Interactive educational elements
- Progressive complexity revelation

Educational Philosophy:
- Every visualization should teach something specific
- Annotations should connect math to intuition
- Progressive disclosure of complexity
- Clear learning objectives for each plot
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

from .style import EDUCATIONAL_COLORS, FONT_SIZES

logger = logging.getLogger(__name__)


class EducationalAnnotator:
    """
    Class for adding educational annotations to matplotlib visualizations.
    
    This class provides methods for adding various types of educational
    content to plots, helping learners understand the concepts being visualized.
    """
    
    def __init__(self, color_scheme: Optional[Dict[str, str]] = None):
        """
        Initialize the educational annotator.
        
        Args:
            color_scheme: Color scheme to use for annotations
        """
        self.color_scheme = color_scheme or {
            'background': EDUCATIONAL_COLORS['neutral_light'],
            'primary': EDUCATIONAL_COLORS['primary_blue'],
            'text': EDUCATIONAL_COLORS['text_dark'],
            'accent': EDUCATIONAL_COLORS['attention_red'],
            'success': EDUCATIONAL_COLORS['success_green']
        }
    
    def add_mathematical_context(self,
                               ax: Axes,
                               concept: str,
                               formula: str,
                               explanation: str,
                               position: str = "top_right") -> None:
        """
        Add mathematical context to a visualization.
        
        Args:
            ax: Axes to annotate
            concept: Mathematical concept name
            formula: Mathematical formula (LaTeX format)
            explanation: Plain text explanation
            position: Position for annotation
        """
        # Create mathematical annotation text
        math_text = f"Mathematical Context: {concept}\n\n"
        math_text += f"Formula: ${formula}$\n\n"
        math_text += f"Explanation:\n{explanation}"
        
        # Position mapping
        positions = {
            "top_right": (0.98, 0.98),
            "top_left": (0.02, 0.98),
            "bottom_right": (0.98, 0.02),
            "bottom_left": (0.02, 0.02)
        }
        
        x, y = positions.get(position, positions["top_right"])
        
        # Add annotation with mathematical styling
        ax.annotate(
            math_text,
            xy=(x, y),
            xycoords='axes fraction',
            bbox={
                'boxstyle': 'round,pad=0.5',
                'facecolor': self.color_scheme['background'],
                'edgecolor': self.color_scheme['primary'],
                'alpha': 0.9,
                'linewidth': 2
            },
            fontsize=FONT_SIZES['annotation'],
            ha='right' if 'right' in position else 'left',
            va='top' if 'top' in position else 'bottom',
            color=self.color_scheme['text'],
            multialignment='left'
        )
    
    def add_performance_insights(self,
                               ax: Axes,
                               metrics: Dict[str, float],
                               interpretations: Dict[str, str],
                               position: str = "top_left") -> None:
        """
        Add performance insights and interpretations.
        
        Args:
            ax: Axes to annotate
            metrics: Performance metrics with values
            interpretations: Interpretation text for each metric
            position: Position for annotation
        """
        # Create insights text
        insights_text = "Performance Insights:\n\n"
        
        for metric, value in metrics.items():
            interpretation = interpretations.get(metric, "")
            insights_text += f"{metric}: {value:.3f}\n"
            if interpretation:
                insights_text += f"  → {interpretation}\n"
            insights_text += "\n"
        
        # Position mapping
        positions = {
            "top_right": (0.98, 0.98),
            "top_left": (0.02, 0.98),
            "bottom_right": (0.98, 0.02),
            "bottom_left": (0.02, 0.02)
        }
        
        x, y = positions.get(position, positions["top_left"])
        
        # Add annotation
        ax.annotate(
            insights_text.strip(),
            xy=(x, y),
            xycoords='axes fraction',
            bbox={
                'boxstyle': 'round,pad=0.5',
                'facecolor': self.color_scheme['success'],
                'edgecolor': self.color_scheme['primary'],
                'alpha': 0.8
            },
            fontsize=FONT_SIZES['annotation'],
            ha='right' if 'right' in position else 'left',
            va='top' if 'top' in position else 'bottom',
            color=self.color_scheme['text'],
            multialignment='left'
        )
    
    def add_learning_objectives(self,
                              fig: Figure,
                              objectives: List[str],
                              title: str = "Learning Objectives") -> None:
        """
        Add learning objectives to the figure.
        
        Args:
            fig: Figure to annotate
            objectives: List of learning objectives
            title: Title for objectives section
        """
        # Create objectives text
        objectives_text = f"{title}:\n\n"
        for i, objective in enumerate(objectives, 1):
            objectives_text += f"{i}. {objective}\n"
        
        # Add as figure text
        fig.text(
            0.02, 0.02,
            objectives_text.strip(),
            bbox={
                'boxstyle': 'round,pad=0.5',
                'facecolor': self.color_scheme['background'],
                'edgecolor': self.color_scheme['accent'],
                'alpha': 0.9
            },
            fontsize=FONT_SIZES['small_text'],
            ha='left',
            va='bottom',
            color=self.color_scheme['text'],
            multialignment='left'
        )
    
    def create_concept_explanation_box(self,
                                     ax: Axes,
                                     concept: str,
                                     explanation: str,
                                     examples: Optional[List[str]] = None,
                                     position: Tuple[float, float] = (0.5, 0.5)) -> None:
        """
        Create a detailed concept explanation box.
        
        Args:
            ax: Axes to add explanation to
            concept: Concept name
            explanation: Detailed explanation
            examples: Optional examples
            position: Position as (x, y) in axes coordinates
        """
        # Build explanation text
        explanation_text = f"Concept: {concept}\n\n{explanation}"
        
        if examples:
            explanation_text += "\n\nExamples:"
            for example in examples:
                explanation_text += f"\n• {example}"
        
        # Create fancy box
        bbox_props = {
            'boxstyle': 'round,pad=0.7',
            'facecolor': self.color_scheme['background'],
            'edgecolor': self.color_scheme['primary'],
            'linewidth': 2,
            'alpha': 0.95
        }
        
        # Add explanation
        ax.text(
            position[0], position[1],
            explanation_text,
            transform=ax.transAxes,
            bbox=bbox_props,
            fontsize=FONT_SIZES['annotation'],
            ha='center',
            va='center',
            color=self.color_scheme['text'],
            multialignment='left',
            wrap=True
        )
    
    def highlight_important_region(self,
                                 ax: Axes,
                                 x_range: Tuple[float, float],
                                 y_range: Tuple[float, float],
                                 label: str,
                                 color: Optional[str] = None) -> None:
        """
        Highlight an important region in the plot.
        
        Args:
            ax: Axes to highlight
            x_range: X coordinate range as (min, max)
            y_range: Y coordinate range as (min, max)
            label: Label for the highlighted region
            color: Color for highlighting (uses accent color if None)
        """
        if color is None:
            color = self.color_scheme['accent']
        
        # Create highlight rectangle
        rect = Rectangle(
            (x_range[0], y_range[0]),
            x_range[1] - x_range[0],
            y_range[1] - y_range[0],
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=0.2
        )
        ax.add_patch(rect)
        
        # Add label
        center_x = (x_range[0] + x_range[1]) / 2
        center_y = (y_range[0] + y_range[1]) / 2
        
        ax.text(
            center_x, center_y,
            label,
            ha='center',
            va='center',
            bbox={
                'boxstyle': 'round,pad=0.3',
                'facecolor': 'white',
                'edgecolor': color,
                'alpha': 0.9
            },
            fontsize=FONT_SIZES['annotation'],
            color=self.color_scheme['text'],
            weight='bold'
        )


# Standalone utility functions for common educational annotations

def add_mathematical_context(ax: Axes,
                           concept: str,
                           formula: str,
                           explanation: str,
                           position: str = "top_right",
                           color_scheme: Optional[Dict[str, str]] = None) -> None:
    """
    Add mathematical context annotation to a plot.
    
    Args:
        ax: Axes to annotate
        concept: Mathematical concept name
        formula: Mathematical formula (LaTeX format)
        explanation: Plain text explanation
        position: Position for annotation
        color_scheme: Color scheme to use
    """
    annotator = EducationalAnnotator(color_scheme)
    annotator.add_mathematical_context(ax, concept, formula, explanation, position)


def add_performance_insights(ax: Axes,
                           metrics: Dict[str, float],
                           interpretations: Dict[str, str],
                           position: str = "top_left",
                           color_scheme: Optional[Dict[str, str]] = None) -> None:
    """
    Add performance insights annotation to a plot.
    
    Args:
        ax: Axes to annotate
        metrics: Performance metrics with values
        interpretations: Interpretation text for each metric
        position: Position for annotation
        color_scheme: Color scheme to use
    """
    annotator = EducationalAnnotator(color_scheme)
    annotator.add_performance_insights(ax, metrics, interpretations, position)


def create_concept_explanation(ax: Axes,
                             concept: str,
                             explanation: str,
                             examples: Optional[List[str]] = None,
                             position: Tuple[float, float] = (0.5, 0.5),
                             color_scheme: Optional[Dict[str, str]] = None) -> None:
    """
    Create a concept explanation box on a plot.
    
    Args:
        ax: Axes to add explanation to
        concept: Concept name
        explanation: Detailed explanation
        examples: Optional examples
        position: Position as (x, y) in axes coordinates
        color_scheme: Color scheme to use
    """
    annotator = EducationalAnnotator(color_scheme)
    annotator.create_concept_explanation_box(ax, concept, explanation, examples, position)


def add_algorithm_steps(ax: Axes,
                       steps: List[str],
                       title: str = "Algorithm Steps",
                       position: str = "bottom_left",
                       color_scheme: Optional[Dict[str, str]] = None) -> None:
    """
    Add algorithm steps explanation to a plot.
    
    Args:
        ax: Axes to annotate
        steps: List of algorithm steps
        title: Title for steps section
        position: Position for annotation
        color_scheme: Color scheme to use
    """
    if color_scheme is None:
        color_scheme = {
            'background': EDUCATIONAL_COLORS['neutral_light'],
            'primary': EDUCATIONAL_COLORS['primary_blue'],
            'text': EDUCATIONAL_COLORS['text_dark']
        }
    
    # Create steps text
    steps_text = f"{title}:\n\n"
    for i, step in enumerate(steps, 1):
        steps_text += f"{i}. {step}\n"
    
    # Position mapping
    positions = {
        "top_right": (0.98, 0.98),
        "top_left": (0.02, 0.98),
        "bottom_right": (0.98, 0.02),
        "bottom_left": (0.02, 0.02)
    }
    
    x, y = positions.get(position, positions["bottom_left"])
    
    # Add annotation
    ax.annotate(
        steps_text.strip(),
        xy=(x, y),
        xycoords='axes fraction',
        bbox={
            'boxstyle': 'round,pad=0.5',
            'facecolor': color_scheme['background'],
            'edgecolor': color_scheme['primary'],
            'alpha': 0.9
        },
        fontsize=FONT_SIZES['annotation'],
        ha='right' if 'right' in position else 'left',
        va='top' if 'top' in position else 'bottom',
        color=color_scheme['text'],
        multialignment='left'
    )


def add_hyperparameter_explanation(ax: Axes,
                                 hyperparams: Dict[str, Any],
                                 explanations: Dict[str, str],
                                 position: str = "top_right",
                                 color_scheme: Optional[Dict[str, str]] = None) -> None:
    """
    Add hyperparameter explanation to a plot.
    
    Args:
        ax: Axes to annotate
        hyperparams: Hyperparameters with values
        explanations: Explanation for each hyperparameter
        position: Position for annotation
        color_scheme: Color scheme to use
    """
    if color_scheme is None:
        color_scheme = {
            'background': EDUCATIONAL_COLORS['neutral_light'],
            'primary': EDUCATIONAL_COLORS['primary_blue'],
            'text': EDUCATIONAL_COLORS['text_dark']
        }
    
    # Create hyperparameter text
    hyperparam_text = "Hyperparameters:\n\n"
    
    for param, value in hyperparams.items():
        explanation = explanations.get(param, "")
        hyperparam_text += f"{param}: {value}\n"
        if explanation:
            hyperparam_text += f"  → {explanation}\n"
        hyperparam_text += "\n"
    
    # Position mapping
    positions = {
        "top_right": (0.98, 0.98),
        "top_left": (0.02, 0.98),
        "bottom_right": (0.98, 0.02),
        "bottom_left": (0.02, 0.02)
    }
    
    x, y = positions.get(position, positions["top_right"])
    
    # Add annotation
    ax.annotate(
        hyperparam_text.strip(),
        xy=(x, y),
        xycoords='axes fraction',
        bbox={
            'boxstyle': 'round,pad=0.5',
            'facecolor': color_scheme['background'],
            'edgecolor': color_scheme['primary'],
            'alpha': 0.9
        },
        fontsize=FONT_SIZES['annotation'],
        ha='right' if 'right' in position else 'left',
        va='top' if 'top' in position else 'bottom',
        color=color_scheme['text'],
        multialignment='left'
    )
