"""
Utility Functions for Educational Visualizations
===============================================

This module provides utility functions used across all visualization components
in the AI-From-Scratch-to-Scale project. These functions handle common tasks
like figure management, path handling, and consistent formatting.

Key Functions:
- save_and_show_plot: Unified figure saving and display logic
- format_axes_for_education: Apply educational formatting to axes
- create_figure_with_theme: Create figures with consistent theming
- add_educational_annotation: Add educational context to plots
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
from typing import Optional, Union, Dict, Any
import logging
import warnings

from .style import EDUCATIONAL_COLORS, FONT_SIZES

logger = logging.getLogger(__name__)


def save_and_show_plot(fig: Figure,
                      filename: Optional[str] = None,
                      save_path: Optional[Union[str, Path]] = None,
                      default_dir: Optional[Path] = None,
                      show: bool = True,
                      close_after: bool = False,
                      dpi: int = 300,
                      bbox_inches: str = 'tight',
                      **kwargs) -> Optional[Path]:
    """
    Save and optionally show a figure with consistent behavior.
    
    This function provides unified logic for saving plots across all
    visualizers, ensuring consistent file handling and display behavior.
    
    Args:
        fig: Figure to save/show
        filename: Name of file to save (if saving)
        save_path: Full path to save file (overrides filename)
        default_dir: Default directory for saving (if not in save_path)
        show: Whether to display the figure
        close_after: Whether to close figure after saving/showing
        dpi: Resolution for saved figure
        bbox_inches: Bounding box setting for saved figure
        **kwargs: Additional arguments for plt.savefig()
        
    Returns:
        Path where figure was saved (if saved), None otherwise
        
    Example:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        saved_path = save_and_show_plot(fig, "my_plot.png", show=True)
    """
    saved_path = None
    
    # Handle saving
    if filename is not None or save_path is not None:
        if save_path is not None:
            # Use provided full path
            save_path = Path(save_path)
        else:
            # Construct path from filename and default directory
            if default_dir is not None:
                save_path = default_dir / filename
            else:
                save_path = Path(filename)
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save figure
        try:
            fig.savefig(
                save_path,
                dpi=dpi,
                bbox_inches=bbox_inches,
                facecolor='white',
                edgecolor='none',
                **kwargs
            )
            saved_path = save_path
            logger.debug(f"Saved figure to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save figure to {save_path}: {e}")
    
    # Handle display
    if show:
        try:
            plt.show()
        except Exception as e:
            logger.warning(f"Failed to display figure: {e}")
    
    # Handle cleanup
    if close_after:
        plt.close(fig)
    
    return saved_path


def format_axes_for_education(ax: Axes, color_scheme: Dict[str, str]) -> None:
    """
    Apply educational formatting to an axes object.
    
    This function ensures consistent styling across all plots by applying
    educational-friendly formatting including colors, fonts, and layout.
    
    Args:
        ax: Axes object to format
        color_scheme: Color scheme dictionary with keys like 'primary', 'text', etc.
    """
    # Set background color
    ax.set_facecolor('white')
    
    # Configure spines (borders)
    for spine in ax.spines.values():
        spine.set_color(color_scheme.get('text', EDUCATIONAL_COLORS['text_dark']))
        spine.set_linewidth(1.0)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Configure ticks
    ax.tick_params(
        colors=color_scheme.get('text', EDUCATIONAL_COLORS['text_dark']),
        labelsize=FONT_SIZES['tick_label']
    )
    
    # Configure grid
    ax.grid(
        True,
        alpha=0.3,
        color=color_scheme.get('text', EDUCATIONAL_COLORS['text_dark']),
        linestyle='-',
        linewidth=0.5
    )
    ax.set_axisbelow(True)  # Grid behind data


def create_figure_with_theme(figsize: tuple = (10, 6),
                           theme: str = "educational") -> tuple:
    """
    Create a figure with consistent theming.
    
    Args:
        figsize: Size of figure as (width, height)
        theme: Theme to apply ("educational" or "professional")
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    if theme == "educational":
        # Apply educational styling
        format_axes_for_education(ax, {
            'text': EDUCATIONAL_COLORS['text_dark'],
            'primary': EDUCATIONAL_COLORS['primary_blue']
        })
    
    return fig, ax


def add_educational_annotation(ax: Axes,
                             text: str,
                             position: str = "top_right",
                             color_scheme: Optional[Dict[str, str]] = None) -> None:
    """
    Add educational annotation to a plot.
    
    Args:
        ax: Axes to annotate
        text: Annotation text
        position: Position preset ("top_right", "top_left", "bottom_right", "bottom_left")
        color_scheme: Color scheme to use for annotation
    """
    if color_scheme is None:
        color_scheme = {
            'background': EDUCATIONAL_COLORS['neutral_light'],
            'primary': EDUCATIONAL_COLORS['primary_blue'],
            'text': EDUCATIONAL_COLORS['text_dark']
        }
    
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
    
    ax.annotate(
        text,
        xy=(x, y),
        xycoords='axes fraction',
        bbox={
            'boxstyle': 'round,pad=0.3',
            'facecolor': color_scheme['background'],
            'edgecolor': color_scheme['primary'],
            'alpha': 0.8
        },
        fontsize=FONT_SIZES['annotation'],
        ha='right' if 'right' in position else 'left',
        va='top' if 'top' in position else 'bottom',
        color=color_scheme['text']
    )


def validate_figure_params(figsize: tuple, 
                          title: str,
                          xlabel: str,
                          ylabel: str) -> Dict[str, Any]:
    """
    Validate and normalize figure parameters.
    
    Args:
        figsize: Figure size tuple
        title: Plot title
        xlabel: X-axis label  
        ylabel: Y-axis label
        
    Returns:
        Dictionary of validated parameters
    """
    # Validate figsize
    if not isinstance(figsize, tuple) or len(figsize) != 2:
        logger.warning(f"Invalid figsize {figsize}, using default (10, 6)")
        figsize = (10, 6)
    
    # Ensure strings
    title = str(title) if title is not None else ""
    xlabel = str(xlabel) if xlabel is not None else ""
    ylabel = str(ylabel) if ylabel is not None else ""
    
    return {
        'figsize': figsize,
        'title': title,
        'xlabel': xlabel,
        'ylabel': ylabel
    }


def get_safe_filename(filename: str) -> str:
    """
    Convert a string to a safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename string
    """
    import re
    # Replace problematic characters with underscores
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove extra spaces and replace with underscores
    safe_name = re.sub(r'\s+', '_', safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    
    return safe_name


def setup_subplot_layout(n_plots: int, 
                        max_cols: int = 3) -> tuple:
    """
    Calculate optimal subplot layout for a given number of plots.
    
    Args:
        n_plots: Number of plots to arrange
        max_cols: Maximum number of columns
        
    Returns:
        Tuple of (rows, cols) for subplot arrangement
    """
    if n_plots <= 0:
        return (1, 1)
    
    if n_plots <= max_cols:
        return (1, n_plots)
    
    cols = min(max_cols, n_plots)
    rows = (n_plots + cols - 1) // cols  # Ceiling division
    
    return (rows, cols)


def ensure_list(item) -> list:
    """
    Ensure an item is a list, wrapping single items.
    
    Args:
        item: Item to ensure is a list
        
    Returns:
        List containing the item(s)
    """
    if item is None:
        return []
    elif isinstance(item, list):
        return item
    elif hasattr(item, '__iter__') and not isinstance(item, str):
        return list(item)
    else:
        return [item]
