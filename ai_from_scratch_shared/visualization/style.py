"""
Styling Framework for Educational Visualizations
===============================================

This module defines consistent styling, color schemes, and themes for all
educational visualizations across the AI-From-Scratch-to-Scale project.

Educational Design Principles:
- High contrast for accessibility
- Colorblind-friendly palettes
- Professional appearance suitable for academic materials
- Consistent typography and spacing
- Mathematical notation support

Color Psychology for Education:
- Blue: Trust, stability, learning concepts
- Green: Success, growth, positive outcomes  
- Red: Attention, errors, important warnings
- Orange: Energy, creativity, engagement
- Purple: Innovation, advanced concepts
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import numpy as np

# =============================================================================
# Educational Color Schemes
# =============================================================================

# Primary educational color palette (colorblind-friendly)
EDUCATIONAL_COLORS = {
    # Core learning colors
    'primary_blue': '#2E86AB',       # Trust, main concepts
    'success_green': '#A23B72',      # Correct predictions, success
    'attention_red': '#F18F01',      # Errors, important points
    'neutral_gray': '#C73E1D',       # Background, secondary info
    
    # Extended palette for complex visualizations
    'deep_blue': '#1B5E87',          # Deep learning concepts
    'light_blue': '#87CEEB',         # Background elements
    'forest_green': '#228B22',       # Nature-inspired, growth
    'warning_orange': '#FF8C00',     # Warnings, attention
    'error_red': '#DC143C',          # Critical errors
    'neutral_light': '#F5F5F5',      # Light backgrounds
    'text_dark': '#2F2F2F',          # Primary text
    'text_medium': '#5F5F5F',        # Secondary text
    
    # Model-specific color schemes
    'perceptron': '#4682B4',         # Steel blue - classic ML
    'mlp': '#32CD32',                # Lime green - growth/learning
    'hopfield': '#FF6347',           # Tomato - energy/dynamics
    'cnn': '#9370DB',                # Medium purple - vision
    'rnn': '#20B2AA',                # Light sea green - sequences
    'transformer': '#FFD700',        # Gold - state-of-the-art
}

# Gradient color maps for heatmaps and continuous data
EDUCATIONAL_COLORMAPS = {
    'learning_progress': ['#FFE4E1', '#FF6347', '#DC143C'],  # Light to dark red
    'accuracy_improvement': ['#F0FFF0', '#32CD32', '#006400'],  # Light to dark green
    'error_magnitude': ['#FFFACD', '#FF8C00', '#FF4500'],   # Light to dark orange
    'neural_activity': ['#F0F8FF', '#4682B4', '#191970'],   # Light to dark blue
    'energy_landscape': ['#E6E6FA', '#9370DB', '#4B0082'],  # Light to dark purple
}

# Class-specific colors for classification tasks
CLASSIFICATION_COLORS = [
    '#FF6B6B',  # Red-ish
    '#4ECDC4',  # Teal
    '#45B7D1',  # Blue
    '#96CEB4',  # Green
    '#FFEAA7',  # Yellow
    '#DDA0DD',  # Plum
    '#F4A460',  # Sandy brown
    '#20B2AA',  # Light sea green
    '#FFB6C1',  # Light pink
    '#87CEFA',  # Light sky blue
]

# =============================================================================
# Figure Specifications
# =============================================================================

FIGURE_SIZES = {
    # Standard sizes for different visualization types
    'default': (10, 6),
    'square': (8, 8),
    'wide': (12, 6),
    'tall': (8, 10),
    
    # Model-specific visualizations
    'confusion_matrix': (8, 6),
    'training_curves': (12, 8),
    'decision_boundary': (10, 8),
    'weight_visualization': (12, 10),
    'pattern_display': (6, 6),
    'energy_landscape': (10, 8),
    
    # Multi-plot layouts
    'subplot_2x2': (12, 10),
    'subplot_1x3': (15, 5),
    'subplot_2x1': (12, 8),
    'comprehensive': (16, 12),
}

# Font specifications for educational materials
FONT_SIZES = {
    'title': 16,
    'subtitle': 14,
    'axis_label': 12,
    'tick_label': 10,
    'legend': 11,
    'annotation': 9,
    'small_text': 8,
}

# Add EDUCATIONAL_STYLE after FONT_SIZES
EDUCATIONAL_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': EDUCATIONAL_COLORS['text_dark'],
    'axes.grid': True,
    'grid.color': EDUCATIONAL_COLORS['neutral_light'],
    'font.family': 'sans-serif',
    'font.size': FONT_SIZES['axis_label'],
    'axes.titlesize': FONT_SIZES['title'],
    'axes.labelsize': FONT_SIZES['axis_label'],
    'xtick.labelsize': FONT_SIZES['tick_label'],
    'ytick.labelsize': FONT_SIZES['tick_label'],
    'legend.fontsize': FONT_SIZES['legend'],
    'text.color': EDUCATIONAL_COLORS['text_dark'],
}

# Line and marker specifications
LINE_STYLES = {
    'solid': '-',
    'dashed': '--', 
    'dotted': ':',
    'dash_dot': '-.',
}

MARKER_STYLES = {
    'circle': 'o',
    'square': 's',
    'triangle': '^',
    'diamond': 'D',
    'cross': 'x',
    'plus': '+',
}

# =============================================================================
# Theme Application Functions
# =============================================================================

def apply_educational_theme() -> None:
    """
    Apply consistent educational theme to matplotlib.
    
    This theme is optimized for:
    - Educational clarity and readability
    - Professional appearance in academic settings
    - Accessibility and colorblind-friendliness
    - Print and digital media compatibility
    """
    plt.style.use('default')  # Start with clean slate
    
    # Configure matplotlib parameters
    plt.rcParams.update({
        # Figure settings
        'figure.facecolor': 'white',
        'figure.edgecolor': 'none',
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': FONT_SIZES['axis_label'],
        'axes.titlesize': FONT_SIZES['title'],
        'axes.labelsize': FONT_SIZES['axis_label'],
        'xtick.labelsize': FONT_SIZES['tick_label'],
        'ytick.labelsize': FONT_SIZES['tick_label'],
        'legend.fontsize': FONT_SIZES['legend'],
        
        # Axes settings
        'axes.facecolor': 'white',
        'axes.edgecolor': EDUCATIONAL_COLORS['text_dark'],
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Grid settings
        'grid.color': EDUCATIONAL_COLORS['neutral_light'],
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,
        
        # Color cycle for multiple series
        'axes.prop_cycle': plt.cycler(color=[
            EDUCATIONAL_COLORS['primary_blue'],
            EDUCATIONAL_COLORS['success_green'], 
            EDUCATIONAL_COLORS['attention_red'],
            EDUCATIONAL_COLORS['warning_orange'],
            EDUCATIONAL_COLORS['deep_blue'],
            EDUCATIONAL_COLORS['forest_green'],
        ]),
        
        # Legend settings
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': False,
        'legend.framealpha': 0.9,
        'legend.edgecolor': EDUCATIONAL_COLORS['neutral_gray'],
        
        # Line and marker settings
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'patch.linewidth': 1.0,
        
        # Text settings
        'text.color': EDUCATIONAL_COLORS['text_dark'],
        'mathtext.default': 'regular',  # Use regular font for math
    })

def apply_professional_theme() -> None:
    """
    Apply professional theme for publication-quality figures.
    
    This theme is optimized for:
    - Academic publication standards
    - High-resolution printing
    - Minimal distractions
    - Clean, professional appearance
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.dpi': 300,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 1.5,
        'grid.alpha': 0.3,
    })

def get_model_color_scheme(model_name: str) -> Dict[str, str]:
    """
    Get color scheme specific to a model type.
    
    Args:
        model_name: Name of the model ('perceptron', 'mlp', 'hopfield', etc.)
        
    Returns:
        Dictionary with color scheme for the model
        
    Example:
        colors = get_model_color_scheme('perceptron')
        plt.plot(x, y, color=colors['primary'])
    """
    base_color = EDUCATIONAL_COLORS.get(model_name.lower(), EDUCATIONAL_COLORS['primary_blue'])
    
    # Generate complementary colors based on base color
    return {
        'primary': base_color,
        'secondary': lighten_color(base_color, 0.3),
        'accent': darken_color(base_color, 0.2),
        'background': lighten_color(base_color, 0.8),
        'text': EDUCATIONAL_COLORS['text_dark'],
        'error': EDUCATIONAL_COLORS['error_red'],
        'success': EDUCATIONAL_COLORS['success_green'],
    }

def create_colormap_from_colors(colors: List[str], name: str = 'custom') -> mcolors.LinearSegmentedColormap:
    """
    Create a custom colormap from a list of colors.
    
    Args:
        colors: List of color hex codes or names
        name: Name for the colormap
        
    Returns:
        Custom matplotlib colormap
    """
    return mcolors.LinearSegmentedColormap.from_list(name, colors)

def lighten_color(color: str, amount: float = 0.3) -> str:
    """
    Lighten a color by mixing it with white.
    
    Args:
        color: Hex color code or color name
        amount: Amount to lighten (0.0 to 1.0)
        
    Returns:
        Lightened color as hex code
    """
    c = mcolors.to_rgb(color)
    lightened = [c[i] + (1 - c[i]) * amount for i in range(3)]
    return mcolors.to_hex(lightened)

def darken_color(color: str, amount: float = 0.3) -> str:
    """
    Darken a color by mixing it with black.
    
    Args:
        color: Hex color code or color name
        amount: Amount to darken (0.0 to 1.0)
        
    Returns:
        Darkened color as hex code
    """
    c = mcolors.to_rgb(color)
    darkened = [c[i] * (1 - amount) for i in range(3)]
    return mcolors.to_hex(darkened)

def get_classification_colors(n_classes: int) -> List[str]:
    """
    Get distinct colors for classification visualization.
    
    Args:
        n_classes: Number of classes to color
        
    Returns:
        List of distinct colors for each class
    """
    if n_classes <= len(CLASSIFICATION_COLORS):
        return CLASSIFICATION_COLORS[:n_classes]
    else:
        # Generate additional colors using color cycle
        base_colors = CLASSIFICATION_COLORS
        additional_colors = plt.cm.Set3(np.linspace(0, 1, n_classes - len(base_colors)))
        return base_colors + [mcolors.to_hex(c) for c in additional_colors]

def setup_educational_style() -> None:
    """
    Set up the complete educational styling environment.
    
    This function should be called once at the beginning of any
    visualization script to ensure consistent styling.
    """
    # Apply base theme
    apply_educational_theme()
    
    # Configure seaborn to complement matplotlib
    sns.set_palette([
        EDUCATIONAL_COLORS['primary_blue'],
        EDUCATIONAL_COLORS['success_green'],
        EDUCATIONAL_COLORS['attention_red'],
        EDUCATIONAL_COLORS['warning_orange'],
        EDUCATIONAL_COLORS['deep_blue'],
        EDUCATIONAL_COLORS['forest_green'],
    ])
    
    # Set default context for educational materials
    sns.set_context("notebook", rc={
        "font.size": FONT_SIZES['axis_label'],
        "axes.titlesize": FONT_SIZES['title'],
        "axes.labelsize": FONT_SIZES['axis_label'],
        "xtick.labelsize": FONT_SIZES['tick_label'],
        "ytick.labelsize": FONT_SIZES['tick_label'],
        "legend.fontsize": FONT_SIZES['legend'],
    })

# Initialize styling on import
setup_educational_style()
