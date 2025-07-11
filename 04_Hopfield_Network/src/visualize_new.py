"""
Hopfield Network Visualizer using Shared Framework
=================================================

This module provides Hopfield Network specific visualizations using the shared
visualization framework. It demonstrates how to extend the BaseVisualizer for
model-specific needs while leveraging common components.

Key Features:
- Pattern visualization (grids and sets)
- Energy landscape analysis
- Convergence dynamics plotting
- Experimental results visualization
- Network state evolution tracking

Educational Focus:
- Energy-based learning principles
- Associative memory concepts
- Pattern storage and retrieval
- Network capacity limitations
- Convergence guarantees
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
    from .config import (
        PATTERN_ON, PATTERN_OFF, PLOTS_DIR, PATTERN_FIGSIZE, 
        ENERGY_FIGSIZE, CONVERGENCE_FIGSIZE
    )
except ImportError:
    # Fall back to absolute imports (when run as script)
    from config import (
        PATTERN_ON, PATTERN_OFF, PLOTS_DIR, PATTERN_FIGSIZE, 
        ENERGY_FIGSIZE, CONVERGENCE_FIGSIZE
    )

logger = logging.getLogger(__name__)


class HopfieldVisualizer(BaseVisualizer):
    """
    Specialized visualizer for Hopfield Networks using shared framework.
    
    This class extends BaseVisualizer to provide Hopfield-specific visualization
    capabilities while maintaining consistency with the shared framework.
    
    Features:
    - Pattern visualization (individual and sets)
    - Energy landscape analysis
    - Convergence dynamics
    - Experimental results plotting
    - Educational annotations for energy-based learning
    """
    
    def __init__(self, default_save_dir: Optional[Path] = None):
        """
        Initialize Hopfield Network visualizer.
        
        Args:
            default_save_dir: Default directory for saving plots
        """
        super().__init__(
            model_name="Hopfield",
            style_theme="educational",
            default_save_dir=default_save_dir or Path(PLOTS_DIR)
        )
        
        # Initialize shared component visualizers
        self.training_curve_viz = TrainingCurveVisualizer()
        self.data_viz = DataDistributionVisualizer()
        self.annotator = EducationalAnnotator(self.colors)
        
        # Hopfield-specific styling
        self.pattern_colors = {
            'on': self.colors['primary'],      # Active neurons
            'off': self.colors['background'],  # Inactive neurons
            'border': self.colors['text'],     # Pattern borders
            'energy_high': self.colors['error'],    # High energy
            'energy_low': self.colors['success'],   # Low energy
        }
        
        logger.debug("Initialized HopfieldVisualizer with shared framework")
    
    def visualize_pattern(self,
                         pattern: np.ndarray,
                         title: str = "Hopfield Pattern",
                         save_path: Optional[Union[str, Path]] = None,
                         show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize a single pattern as a 2D grid.
        
        Args:
            pattern: Pattern array to visualize
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            Tuple of (figure, axes)
        """
        # Determine pattern dimensions
        size = int(np.sqrt(len(pattern)))
        if size * size != len(pattern):
            raise ValueError(f"Pattern length {len(pattern)} is not a perfect square")
        
        # Reshape to 2D
        pattern_2d = pattern.reshape(size, size)
        
        # Create figure
        fig, ax = self.create_figure(figsize='pattern_display')
        
        # Create custom colormap for binary patterns
        colors = [self.pattern_colors['off'], self.pattern_colors['on']]
        cmap = ListedColormap(colors)
        
        # Display pattern
        im = ax.imshow(
            pattern_2d,
            cmap=cmap,
            interpolation='nearest',
            vmin=-1,
            vmax=1
        )
        
        # Styling
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add grid for clarity
        for i in range(size + 1):
            ax.axhline(i - 0.5, color=self.pattern_colors['border'], linewidth=0.5)
            ax.axvline(i - 0.5, color=self.pattern_colors['border'], linewidth=0.5)
        
        # Add educational annotation
        self.add_educational_annotation(
            ax,
            f"Pattern: {size}×{size} grid\n"
            f"Active neurons: {np.sum(pattern > 0)}\n"
            f"Inactive neurons: {np.sum(pattern < 0)}",
            position="bottom_right"
        )
        
        # Add mathematical context
        add_mathematical_context(
            ax,
            concept="Binary Pattern Representation",
            formula=r"x_i \in \{-1, +1\}",
            explanation="Each neuron can be in one of two states: active (+1) or inactive (-1)."
        )
        
        # Save and show
        self.save_and_show(fig, save_path=save_path, show=show)
        
        return fig, ax
    
    def visualize_pattern_set(self,
                             patterns: Dict[str, np.ndarray],
                             title: str = "Hopfield Pattern Set",
                             save_path: Optional[Union[str, Path]] = None,
                             show: bool = True) -> Tuple[plt.Figure, np.ndarray]:
        """
        Visualize a set of patterns in a grid layout.
        
        Args:
            patterns: Dictionary of pattern name -> pattern array
            title: Overall title for the figure
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            Tuple of (figure, axes array)
        """
        n_patterns = len(patterns)
        if n_patterns == 0:
            raise ValueError("No patterns provided")
        
        # Calculate grid layout
        cols = min(4, n_patterns)
        rows = (n_patterns + cols - 1) // cols
        
        # Create figure with subplots
        fig, axes = self.create_figure(
            figsize=(4 * cols, 4 * rows),
            subplots=(rows, cols)
        )
        
        # Ensure axes is iterable
        if n_patterns == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten()
        
        # Plot each pattern
        pattern_items = list(patterns.items())
        for i, (name, pattern) in enumerate(pattern_items):
            ax = axes[i]
            
            # Determine pattern dimensions
            size = int(np.sqrt(len(pattern)))
            pattern_2d = pattern.reshape(size, size)
            
            # Create colormap
            colors = [self.pattern_colors['off'], self.pattern_colors['on']]
            cmap = ListedColormap(colors)
            
            # Display pattern
            ax.imshow(
                pattern_2d,
                cmap=cmap,
                interpolation='nearest',
                vmin=-1,
                vmax=1
            )
            
            # Styling
            ax.set_title(name, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add grid
            for j in range(size + 1):
                ax.axhline(j - 0.5, color=self.pattern_colors['border'], linewidth=0.5)
                ax.axvline(j - 0.5, color=self.pattern_colors['border'], linewidth=0.5)
        
        # Hide unused subplots
        for i in range(n_patterns, len(axes)):
            axes[i].set_visible(False)
        
        # Overall title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Add educational annotation to figure
        fig.text(
            0.02, 0.02,
            f"Stored {n_patterns} patterns for associative memory.\n"
            f"Each pattern represents a stable state in the energy landscape.",
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
    
    def visualize_energy_landscape(self,
                                  stored_patterns: List[np.ndarray],
                                  weights: np.ndarray,
                                  title: str = "Energy Landscape Analysis",
                                  save_path: Optional[Union[str, Path]] = None,
                                  show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize the energy landscape of the Hopfield network.
        
        Args:
            stored_patterns: List of stored patterns
            weights: Network weight matrix
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            Tuple of (figure, axes)
        """
        if not stored_patterns:
            raise ValueError("No stored patterns provided")
        
        # Calculate energies for stored patterns
        energies = []
        pattern_names = []
        
        for i, pattern in enumerate(stored_patterns):
            # Calculate energy: E = -0.5 * x^T * W * x
            energy = -0.5 * np.dot(pattern.T, np.dot(weights, pattern))
            energies.append(energy)
            pattern_names.append(f"Pattern {i+1}")
        
        # Create figure
        fig, ax = self.create_figure(figsize='energy_landscape')
        
        # Create bar plot of energies
        bars = ax.bar(
            range(len(energies)),
            energies,
            color=self.pattern_colors['energy_low'],
            edgecolor=self.pattern_colors['border'],
            alpha=0.8
        )
        
        # Styling
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel("Stored Patterns")
        ax.set_ylabel("Energy")
        ax.set_xticks(range(len(energies)))
        ax.set_xticklabels(pattern_names, rotation=45)
        
        # Add energy values on bars
        for bar, energy in zip(bars, energies):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + (max(energies) - min(energies)) * 0.01,
                f"{energy:.2f}",
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        # Add mathematical context
        add_mathematical_context(
            ax,
            concept="Hopfield Energy Function",
            formula=r"E = -\frac{1}{2} \sum_{i,j} w_{ij} x_i x_j",
            explanation="Lower energy states are more stable. Stored patterns should be local minima."
        )
        
        # Add educational annotation
        self.add_educational_annotation(
            ax,
            "Stored patterns should have low energy values.\n"
            "The network will converge to these stable states\n"
            "from nearby initial conditions.",
            position="top_right"
        )
        
        plt.tight_layout()
        
        # Save and show
        self.save_and_show(fig, save_path=save_path, show=show)
        
        return fig, ax
    
    def plot_convergence_analysis(self,
                                 convergence_data: Dict[str, List[float]],
                                 title: str = "Convergence Dynamics Analysis",
                                 save_path: Optional[Union[str, Path]] = None,
                                 show: bool = True) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot comprehensive convergence analysis.
        
        Args:
            convergence_data: Dictionary containing convergence statistics
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            Tuple of (figure, axes array)
        """
        # Create 2x2 subplot layout
        fig, axes = self.create_figure(figsize=(12, 10), subplots=(2, 2))
        
        # Plot 1: Energy histories
        ax1 = axes[0, 0]
        if 'energy_histories' in convergence_data:
            for i, energy_hist in enumerate(convergence_data['energy_histories'][:5]):  # Show first 5
                ax1.plot(energy_hist, alpha=0.7, label=f'Trial {i+1}')
            ax1.set_title("Energy Convergence Traces")
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Energy")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Convergence steps distribution
        ax2 = axes[0, 1]
        if 'convergence_steps' in convergence_data:
            ax2.hist(
                convergence_data['convergence_steps'],
                bins=20,
                color=self.colors['primary'],
                alpha=0.7,
                edgecolor='black'
            )
            ax2.set_title("Convergence Steps Distribution")
            ax2.set_xlabel("Steps to Convergence")
            ax2.set_ylabel("Frequency")
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Energy decrease analysis
        ax3 = axes[1, 0]
        if 'energy_decreases' in convergence_data:
            ax3.scatter(
                range(len(convergence_data['energy_decreases'])),
                convergence_data['energy_decreases'],
                color=self.colors['success'],
                alpha=0.6
            )
            ax3.set_title("Energy Decrease per Trial")
            ax3.set_xlabel("Trial")
            ax3.set_ylabel("Energy Decrease")
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Overlap improvement
        ax4 = axes[1, 1]
        if 'initial_overlaps' in convergence_data and 'final_overlaps' in convergence_data:
            initial = convergence_data['initial_overlaps']
            final = convergence_data['final_overlaps']
            improvements = [f - i for i, f in zip(initial, final)]
            
            ax4.scatter(initial, final, color=self.colors['primary'], alpha=0.6)
            ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No improvement')
            ax4.set_title("Overlap Improvement")
            ax4.set_xlabel("Initial Overlap")
            ax4.set_ylabel("Final Overlap")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Add educational annotation
        fig.text(
            0.02, 0.02,
            "Convergence analysis shows the dynamics of energy minimization.\n"
            "Successful retrieval requires convergence to stored pattern states.",
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
    
    def plot_capacity_analysis(self,
                              capacity_results: Dict[int, Dict[str, float]],
                              network_size: int,
                              title: str = "Storage Capacity Analysis",
                              save_path: Optional[Union[str, Path]] = None,
                              show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot network capacity analysis results.
        
        Args:
            capacity_results: Results from capacity experiments
            network_size: Size of the network
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            Tuple of (figure, axes)
        """
        # Extract data
        pattern_counts = list(capacity_results.keys())
        success_rates = [results['success_rate'] for results in capacity_results.values()]
        error_bars = [results.get('success_rate_std', 0) for results in capacity_results.values()]
        theoretical_ratios = [results['theoretical_ratio'] for results in capacity_results.values()]
        
        # Create figure
        fig, ax = self.create_figure(figsize='training_curves')
        
        # Plot success rate vs number of patterns
        ax.errorbar(
            pattern_counts,
            success_rates,
            yerr=error_bars,
            marker='o',
            linewidth=2,
            markersize=8,
            color=self.colors['primary'],
            capsize=5,
            label='Experimental Results'
        )
        
        # Add theoretical capacity line
        theoretical_capacity = int(0.15 * network_size)
        ax.axvline(
            theoretical_capacity,
            color=self.colors['error'],
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            label=f'Theoretical Capacity (~{theoretical_capacity})'
        )
        
        # Styling
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel("Number of Stored Patterns")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add performance insights
        insights = {
            "Network Size": network_size,
            "Theoretical Capacity": theoretical_capacity,
            "Max Tested Patterns": max(pattern_counts),
            "Best Success Rate": max(success_rates)
        }
        
        interpretations = {
            "Network Size": "Total number of neurons in the network",
            "Theoretical Capacity": "≈0.15 × N patterns can be stored reliably",
            "Max Tested Patterns": "Highest number of patterns tested",
            "Best Success Rate": "Best retrieval performance achieved"
        }
        
        add_performance_insights(ax, insights, interpretations, position="top_right")
        
        # Add mathematical context
        add_mathematical_context(
            ax,
            concept="Hopfield Capacity",
            formula=r"C \approx 0.15 \times N",
            explanation="Network capacity scales linearly with size but with low coefficient."
        )
        
        # Save and show
        self.save_and_show(fig, save_path=save_path, show=show)
        
        return fig, ax
    
    def plot_noise_robustness(self,
                             noise_results: Dict[float, Dict[str, float]],
                             title: str = "Noise Robustness Analysis",
                             save_path: Optional[Union[str, Path]] = None,
                             show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot noise robustness analysis results.
        
        Args:
            noise_results: Results from noise robustness experiments
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            Tuple of (figure, axes)
        """
        # Extract data
        noise_levels = list(noise_results.keys())
        success_rates = [results['success_rate'] for results in noise_results.values()]
        error_bars = [results.get('success_rate_std', 0) for results in noise_results.values()]
        
        # Create figure
        fig, ax = self.create_figure(figsize='training_curves')
        
        # Plot success rate vs noise level
        ax.errorbar(
            noise_levels,
            success_rates,
            yerr=error_bars,
            marker='o',
            linewidth=2,
            markersize=8,
            color=self.colors['primary'],
            capsize=5,
            label='Retrieval Success Rate'
        )
        
        # Add 50% success threshold line
        ax.axhline(
            0.5,
            color=self.colors['error'],
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            label='50% Success Threshold'
        )
        
        # Styling
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel("Noise Level")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add educational annotation
        self.add_educational_annotation(
            ax,
            "Hopfield networks can recover patterns from noisy input.\n"
            "Performance degrades as noise level increases.\n"
            "Associative memory provides error correction.",
            position="top_right"
        )
        
        # Add mathematical context
        add_mathematical_context(
            ax,
            concept="Error Correction",
            formula=r"\text{overlap} = \frac{\mathbf{x} \cdot \mathbf{y}}{|\mathbf{x}||\mathbf{y}|}",
            explanation="Higher overlap between retrieved and original patterns indicates better error correction."
        )
        
        # Save and show
        self.save_and_show(fig, save_path=save_path, show=show)
        
        return fig, ax


# Backwards compatibility functions that maintain existing API
def display_pattern(pattern: np.ndarray, title: str = "Pattern") -> None:
    """Console display of pattern (backwards compatibility)."""
    size = int(np.sqrt(len(pattern)))
    pattern_2d = pattern.reshape(size, size)
    
    print(f"\n{title}:")
    print("-" * (size * 2 + 1))
    for row in pattern_2d:
        print("|", end="")
        for val in row:
            print("█" if val > 0 else " ", end="")
        print("|")
    print("-" * (size * 2 + 1))


def visualize_pattern_set(patterns: Dict[str, np.ndarray],
                         title: str = "Pattern Set",
                         save_path: Optional[str] = None,
                         show: bool = True) -> None:
    """Backwards compatible pattern set visualization."""
    viz = HopfieldVisualizer()
    viz.visualize_pattern_set(patterns, title, save_path, show)
    viz.cleanup_figures()


def visualize_energy_landscape(stored_patterns: List[np.ndarray],
                              weights: np.ndarray,
                              save_path: Optional[str] = None,
                              show: bool = True) -> None:
    """Backwards compatible energy landscape visualization."""
    viz = HopfieldVisualizer()
    viz.visualize_energy_landscape(stored_patterns, weights, save_path=save_path, show=show)
    viz.cleanup_figures()


def plot_convergence_statistics(convergence_results: Dict[str, List[float]],
                               title: str = "Convergence Statistics",
                               save_path: Optional[str] = None,
                               show: bool = True) -> None:
    """Backwards compatible convergence statistics plotting."""
    viz = HopfieldVisualizer()
    viz.plot_convergence_analysis(convergence_results, title, save_path, show)
    viz.cleanup_figures()


def plot_capacity_results(capacity_results: Dict[int, Dict[str, float]],
                         network_size: int,
                         wandb_visualizer=None,
                         save_path: Optional[str] = None,
                         show: bool = True) -> None:
    """Backwards compatible capacity results plotting."""
    viz = HopfieldVisualizer()
    viz.plot_capacity_analysis(capacity_results, network_size, save_path=save_path, show=show)
    viz.cleanup_figures()


def plot_noise_robustness(noise_results: Dict[float, Dict[str, float]],
                         pattern_type: str,
                         title: str = "Noise Robustness",
                         save_path: Optional[str] = None,
                         show: bool = True) -> None:
    """Backwards compatible noise robustness plotting."""
    viz = HopfieldVisualizer()
    viz.plot_noise_robustness(noise_results, f"{title} - {pattern_type}", save_path, show)
    viz.cleanup_figures()


def create_comprehensive_comparison(experiment_results: Dict[str, Any]) -> None:
    """Backwards compatible comprehensive comparison plotting."""
    viz = HopfieldVisualizer()
    
    # Create comprehensive visualization combining all results
    fig, axes = viz.create_figure(figsize=(16, 12), subplots=(2, 2))
    
    # Plot different experimental results in subplots
    if 'capacity' in experiment_results:
        # This would be implemented based on available data
        pass
    
    fig.suptitle("Comprehensive Hopfield Network Analysis", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    viz.save_and_show(fig, "comprehensive_analysis.png", show=True)
    viz.cleanup_figures()


# Legacy functions for existing imports
def plot_spatial_invariance_results(*args, **kwargs):
    """Placeholder for spatial invariance results."""
    logger.warning("plot_spatial_invariance_results not yet implemented with shared framework")
    pass
