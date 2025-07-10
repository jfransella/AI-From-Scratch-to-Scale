"""
Hopfield Network Visualization Module
====================================

This module consolidates ALL visualization functionality for the Hopfield Network,
following the single responsibility principle from the AI-From-Scratch-to-Scale
coding guidelines.

Educational Purpose:
- Pattern visualization (console and matplotlib)
- Energy landscape analysis
- Convergence dynamics
- Experimental results plotting
- Network state comparisons

Visualization Types:
1. Pattern Display (2D grids, pattern sets)
2. Energy Analysis (landscapes, convergence)
3. Experimental Results (capacity, noise robustness)
4. Network Dynamics (state evolution, statistics)
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

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


# ============================================================================
# PATTERN VISUALIZATION
# ============================================================================

def display_pattern(pattern: np.ndarray, title: str = "Pattern") -> None:
    """
    Display a flattened bipolar pattern as a 2D grid on the console.

    Args:
        pattern: A flattened 1D NumPy array with bipolar (-1, 1) values
        title: A title to print above the pattern
        
    Raises:
        ValueError: If pattern length is not a perfect square
        
    Educational Focus:
        Provides immediate visual feedback for pattern inspection without
        requiring matplotlib, useful for debugging and quick analysis.
    """
    # Assuming a square pattern, calculate the side length
    side_length = int(np.sqrt(len(pattern)))
    if side_length * side_length != len(pattern):
        raise ValueError("Pattern length must be a perfect square.")

    # Reshape the pattern into a 2D grid
    grid = pattern.reshape((side_length, side_length))

    print(f"--- {title} ---")
    for row in grid:
        for pixel in row:
            # Use a block character for 'on' pixels and a space for 'off' pixels
            char = '█' if pixel == PATTERN_ON else ' '
            print(char, end=' ')
        print()  # Newline at the end of the row
    print("-" * (side_length * 2 + 3))


def visualize_pattern(pattern: np.ndarray, title: str = "Pattern", 
                     save_path: Optional[str] = None, show: bool = True,
                     pattern_size: Optional[Tuple[int, int]] = None) -> matplotlib.figure.Figure:
    """
    Visualize a single binary pattern as a 2D heatmap.
    
    Args:
        pattern: Binary pattern as 1D array
        title: Title for the plot
        save_path: Optional path to save the figure
        show: Whether to display the plot
        pattern_size: Optional (height, width) tuple, auto-detected if None
        
    Returns:
        The matplotlib figure object
        
    Educational Focus:
        Clear visualization of stored patterns showing the spatial structure
        that the Hopfield network must learn to associate and recall.
    """
    # Determine pattern dimensions
    if pattern_size is None:
        side_length = int(np.sqrt(len(pattern)))
        if side_length * side_length != len(pattern):
            raise ValueError("Pattern must be square or pattern_size must be provided")
        height, width = side_length, side_length
    else:
        height, width = pattern_size
        if height * width != len(pattern):
            raise ValueError(f"Pattern size {len(pattern)} doesn't match dimensions {height}x{width}")
    
    # Reshape pattern to 2D grid
    grid = pattern.reshape((height, width))
    
    # Create visualization
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='RdBu', vmin=-1, vmax=1)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar(label='Neuron State')
    
    # Add grid for clarity
    plt.grid(True, alpha=0.3)
    plt.xticks(range(width))
    plt.yticks(range(height))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Pattern visualization saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_pattern_set(patterns: Dict[str, np.ndarray], 
                         title: str = "Pattern Set",
                         save_path: Optional[str] = None,
                         show: bool = True,
                         pattern_size: Optional[Tuple[int, int]] = None) -> matplotlib.figure.Figure:
    """
    Visualize a collection of patterns in a grid layout.
    
    Args:
        patterns: Dictionary of pattern_name -> pattern_array
        title: Overall title for the plot
        save_path: Optional path to save the figure
        show: Whether to display the plot
        pattern_size: Optional (height, width) tuple for each pattern
        
    Returns:
        The matplotlib figure object
        
    Educational Focus:
        Shows the diversity and structure of patterns that will be stored
        in the network, helping understand capacity and interference issues.
    """
    if not patterns:
        logger.warning("No patterns provided for visualization")
        return None
    
    # Calculate grid layout
    num_patterns = len(patterns)
    cols = min(4, num_patterns)  # Maximum 4 columns
    rows = (num_patterns + cols - 1) // cols
    
    # Create subplot grid
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    # Plot each pattern
    for idx, (name, pattern) in enumerate(patterns.items()):
        row, col = idx // cols, idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        
        # Determine pattern dimensions
        if pattern_size is None:
            side_length = int(np.sqrt(len(pattern)))
            height, width = side_length, side_length
        else:
            height, width = pattern_size
        
        # Reshape and display
        grid = pattern.reshape((height, width))
        im = ax.imshow(grid, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(name, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(num_patterns, rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Pattern set visualization saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# ENERGY LANDSCAPE VISUALIZATION
# ============================================================================

def visualize_energy_landscape(stored_patterns: List[np.ndarray],
                              weights: np.ndarray,
                              num_samples: int = 1000,
                              title: str = "Energy Landscape",
                              save_path: Optional[str] = None,
                              show: bool = True) -> matplotlib.figure.Figure:
    """
    Visualize the energy landscape of the Hopfield network.
    
    Args:
        stored_patterns: List of stored patterns
        weights: Network weight matrix
        num_samples: Number of random states to sample
        title: Title for the plot
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        The matplotlib figure object
        
    Educational Focus:
        Demonstrates how stored patterns create low-energy attractors,
        illustrating the energy-based paradigm and Lyapunov function.
    """
    def calculate_energy(state: np.ndarray) -> float:
        """Calculate energy of a state."""
        return -0.5 * np.sum(weights * np.outer(state, state))
    
    if not stored_patterns:
        logger.warning("No stored patterns provided for energy landscape")
        return None
    
    network_size = len(stored_patterns[0])
    
    # Sample random states and calculate energies
    random_energies = []
    for _ in range(num_samples):
        random_state = np.random.choice([PATTERN_OFF, PATTERN_ON], size=network_size)
        energy = calculate_energy(random_state)
        random_energies.append(energy)
    
    # Calculate energies of stored patterns
    stored_energies = []
    for pattern in stored_patterns:
        energy = calculate_energy(pattern)
        stored_energies.append(energy)
    
    # Create visualization
    fig = plt.figure(figsize=ENERGY_FIGSIZE)
    
    # Plot random state energies as histogram
    plt.hist(random_energies, bins=50, alpha=0.7, density=True, 
            label='Random States', color='lightblue')
    
    # Plot stored pattern energies as vertical lines
    for i, energy in enumerate(stored_energies):
        plt.axvline(energy, color='red', linestyle='--', alpha=0.8,
                   label='Stored Patterns' if i == 0 else "")
    
    plt.xlabel('Energy', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add network info
    info_text = f"Network Size: {network_size}\\nStored Patterns: {len(stored_patterns)}"
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Energy landscape saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_convergence(energy_history: List[float], 
                         title: str = "Energy Convergence",
                         save_path: Optional[str] = None,
                         show: bool = True) -> matplotlib.figure.Figure:
    """
    Visualize energy convergence during pattern retrieval.
    
    Args:
        energy_history: List of energy values over iterations
        title: Title for the plot
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        The matplotlib figure object
        
    Educational Focus:
        Demonstrates the Lyapunov property - energy always decreases
        during asynchronous updates, guaranteeing convergence.
    """
    fig = plt.figure(figsize=CONVERGENCE_FIGSIZE)
    
    plt.plot(energy_history, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Highlight energy decrease
    if len(energy_history) > 1:
        energy_decrease = energy_history[0] - energy_history[-1]
        plt.text(0.02, 0.98, f'Energy Decrease: {energy_decrease:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Convergence plot saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# EXPERIMENTAL RESULTS VISUALIZATION
# ============================================================================

def plot_capacity_results(capacity_results: Dict[int, Dict[str, float]],
                         network_size: int,
                         title: str = "Storage Capacity Analysis",
                         save_path: Optional[str] = None,
                         show: bool = True) -> matplotlib.figure.Figure:
    """
    Plot storage capacity experiment results.
    
    Args:
        capacity_results: Results from capacity experiment
        network_size: Size of the network
        title: Title for the plot
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        The matplotlib figure object
        
    Educational Focus:
        Demonstrates the theoretical capacity limit (~0.15 * N) and
        performance degradation beyond this limit.
    """
    pattern_counts = list(capacity_results.keys())
    success_rates = [capacity_results[n]['success_rate'] for n in pattern_counts]
    error_bars = [capacity_results[n]['success_rate_std'] for n in pattern_counts]
    theoretical_ratios = [capacity_results[n]['theoretical_ratio'] for n in pattern_counts]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Success rate vs number of patterns
    ax1.errorbar(pattern_counts, success_rates, yerr=error_bars, 
                marker='o', linewidth=2, capsize=5)
    ax1.axvline(0.15 * network_size, color='red', linestyle='--', 
               label=f'Theoretical Capacity (~{int(0.15 * network_size)})')
    ax1.set_xlabel('Number of Stored Patterns')
    ax1.set_ylabel('Retrieval Success Rate')
    ax1.set_title('Storage Capacity vs. Performance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Success rate vs theoretical capacity ratio
    ax2.plot(theoretical_ratios, success_rates, 'bo-', linewidth=2)
    ax2.axvline(1.0, color='red', linestyle='--', label='Theoretical Limit')
    ax2.set_xlabel('Ratio to Theoretical Capacity')
    ax2.set_ylabel('Retrieval Success Rate')
    ax2.set_title('Performance vs. Theoretical Capacity')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Capacity results plot saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_noise_robustness(noise_results: Dict[float, Dict[str, float]], 
                         pattern_type: str,
                         title: str = "Noise Robustness Analysis",
                         save_path: Optional[str] = None,
                         show: bool = True) -> matplotlib.figure.Figure:
    """
    Plot noise robustness experiment results.
    
    Args:
        noise_results: Results from noise experiment
        pattern_type: Type of patterns tested
        title: Title for the plot
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        The matplotlib figure object
        
    Educational Focus:
        Shows error correction capabilities and robustness to
        partial or corrupted input information.
    """
    noise_levels = list(noise_results.keys())
    success_rates = [noise_results[n]['success_rate'] for n in noise_levels]
    success_errors = [noise_results[n]['success_rate_std'] for n in noise_levels]
    overlap_improvements = [noise_results[n]['avg_overlap_improvement'] for n in noise_levels]
    overlap_errors = [noise_results[n]['overlap_improvement_std'] for n in noise_levels]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Success rate vs noise level
    ax1.errorbar(noise_levels, success_rates, yerr=success_errors,
                marker='o', linewidth=2, capsize=5, color='blue')
    ax1.set_xlabel('Noise Level (Fraction of Bits Flipped)')
    ax1.set_ylabel('Retrieval Success Rate')
    ax1.set_title(f'Noise Robustness - {pattern_type}')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Overlap improvement vs noise level
    ax2.errorbar(noise_levels, overlap_improvements, yerr=overlap_errors,
                marker='s', linewidth=2, capsize=5, color='green')
    ax2.set_xlabel('Noise Level (Fraction of Bits Flipped)')
    ax2.set_ylabel('Average Overlap Improvement')
    ax2.set_title('Pattern Recovery Quality')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Noise robustness plot saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_convergence_statistics(convergence_results: Dict[str, List[float]],
                               title: str = "Convergence Dynamics Analysis",
                               save_path: Optional[str] = None,
                               show: bool = True) -> matplotlib.figure.Figure:
    """
    Plot convergence dynamics statistics.
    
    Args:
        convergence_results: Results from convergence experiment
        title: Title for the plot
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        The matplotlib figure object
        
    Educational Focus:
        Analyzes convergence behavior, energy minimization,
        and relationship between initial conditions and outcomes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Energy decrease distribution
    axes[0, 0].hist(convergence_results['energy_decreases'], bins=20, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('Energy Decrease')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Energy Decreases')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Convergence steps distribution
    axes[0, 1].hist(convergence_results['convergence_steps'], bins=20, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Convergence Steps')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Convergence Times')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Overlap improvement scatter
    initial_overlaps = convergence_results['initial_overlaps']
    final_overlaps = convergence_results['final_overlaps']
    overlap_improvements = [f - i for i, f in zip(initial_overlaps, final_overlaps)]
    
    axes[1, 0].scatter(initial_overlaps, overlap_improvements, alpha=0.6)
    axes[1, 0].set_xlabel('Initial Overlap')
    axes[1, 0].set_ylabel('Overlap Improvement')
    axes[1, 0].set_title('Pattern Recovery vs. Initial Quality')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Example energy trajectory
    if convergence_results['energy_histories']:
        example_history = convergence_results['energy_histories'][0]
        axes[1, 1].plot(example_history, 'b-', linewidth=2, marker='o', markersize=4)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Energy')
        axes[1, 1].set_title('Example Energy Convergence')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Convergence statistics plot saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# COMPARISON AND ANALYSIS VISUALIZATION
# ============================================================================

def create_comprehensive_comparison(experiment_results: Dict[str, Any],
                                  title: str = "Comprehensive Experiment Analysis",
                                  save_path: Optional[str] = None,
                                  show: bool = True) -> matplotlib.figure.Figure:
    """
    Create comprehensive comparison visualization across all experiments.
    
    Args:
        experiment_results: Dictionary containing results from all experiments
        title: Title for the overall figure
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        The matplotlib figure object
        
    Educational Focus:
        Provides holistic view of network performance across different
        experimental conditions and metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Capacity Analysis
    if 'capacity' in experiment_results:
        capacity_data = experiment_results['capacity']
        network_sizes = []
        success_rates = []
        
        for key, value in capacity_data.items():
            if isinstance(key, str) and key.endswith('_success_rate'):
                try:
                    size = int(key.split('_')[0])
                    network_sizes.append(size)
                    success_rates.append(value)
                except (ValueError, IndexError):
                    continue
        
        if network_sizes and success_rates:
            sorted_data = sorted(zip(network_sizes, success_rates))
            network_sizes, success_rates = zip(*sorted_data)
            
            axes[0, 0].plot(network_sizes, success_rates, 'bo-', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('Number of Stored Patterns')
            axes[0, 0].set_ylabel('Success Rate')
            axes[0, 0].set_title('Storage Capacity Analysis')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No capacity data available', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
    else:
        axes[0, 0].text(0.5, 0.5, 'No capacity data available', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
    
    # Plot 2: Noise Robustness
    if 'noise_robustness' in experiment_results:
        noise_data = experiment_results.get('noise_robustness', {})
        noise_levels = []
        success_rates = []
        
        for key, value in noise_data.items():
            if isinstance(key, str) and 'success_rate_' in key and not key.endswith('_std'):
                try:
                    noise_level = float(key.split('success_rate_')[1])
                    noise_levels.append(noise_level)
                    success_rates.append(value)
                except (ValueError, IndexError):
                    continue
        
        if noise_levels and success_rates:
            sorted_data = sorted(zip(noise_levels, success_rates))
            noise_levels, success_rates = zip(*sorted_data)
            
            axes[0, 1].plot(noise_levels, success_rates, 'go-', linewidth=2, markersize=6)
            axes[0, 1].set_xlabel('Noise Level')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_title('Noise Robustness Analysis')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(-0.05, 1.05)
        else:
            axes[0, 1].text(0.5, 0.5, 'No noise data available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
    else:
        axes[0, 1].text(0.5, 0.5, 'No noise data available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # Plot 3: Convergence Dynamics
    if 'convergence' in experiment_results:
        conv_data = experiment_results['convergence']
        
        metrics = ['Mean Steps', 'Energy Decrease', 'Final Overlap']
        values = [
            conv_data.get('convergence_steps_mean', 0),
            conv_data.get('energy_decreases_mean', 0) / 10,  # Scale for visibility
            conv_data.get('final_overlaps_mean', 0)
        ]
        
        bars = axes[1, 0].bar(metrics, values, color=['blue', 'red', 'green'], alpha=0.7)
        axes[1, 0].set_ylabel('Normalized Values')
        axes[1, 0].set_title('Convergence Dynamics Summary')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
    else:
        axes[1, 0].text(0.5, 0.5, 'No convergence data available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Plot 4: Overall Performance Summary
    experiments = []
    performance_scores = []
    
    if 'basic_training' in experiment_results:
        experiments.append('Basic\\nTraining')
        basic_score = experiment_results['basic_training'].get('perfect_recall_rate', 0)
        performance_scores.append(basic_score)
    
    if 'capacity' in experiment_results:
        experiments.append('Capacity\\nAnalysis')
        capacity_scores = [v for k, v in experiment_results['capacity'].items() 
                         if isinstance(k, str) and k.endswith('_success_rate')]
        capacity_score = np.mean(capacity_scores) if capacity_scores else 0
        performance_scores.append(capacity_score)
    
    if experiments and performance_scores:
        bars = axes[1, 1].bar(experiments, performance_scores, 
                            color=['skyblue', 'lightcoral'][:len(experiments)],
                            alpha=0.8)
        axes[1, 1].set_ylabel('Performance Score')
        axes[1, 1].set_title('Overall Experiment Performance')
        axes[1, 1].set_ylim(0, 1.1)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, performance_scores):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'No performance data available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comprehensive comparison saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# SPATIAL INVARIANCE VISUALIZATION
# ============================================================================

def plot_spatial_invariance_results(results: List[Dict], labels: List[str], overlaps: List[float]) -> None:
    """
    Create visualization showing spatial invariance failure.
    
    Args:
        results: List of spatial invariance test results
        labels: List of shift labels
        overlaps: List of overlap scores
        
    Educational Focus:
        Visualizes why Hopfield networks fail with spatial translations,
        demonstrating the motivation for convolutional architectures.
    """
    n_patterns = len(results)
    size = int(np.sqrt(len(results[0]['input_pattern'])))
    
    # Create subplot grid
    fig, axes = plt.subplots(3, n_patterns, figsize=(2*n_patterns, 6))
    fig.suptitle('Hopfield Network Spatial Invariance Limitation', fontsize=16)
    
    for i, (result, label) in enumerate(zip(results, labels)):
        # Input pattern
        axes[0, i].imshow(result['input_pattern'].reshape(size, size), 
                         cmap='RdBu', vmin=-1, vmax=1)
        axes[0, i].set_title(f'Input: {label}')
        axes[0, i].axis('off')
        
        # Retrieved pattern
        axes[1, i].imshow(result['retrieved_pattern'].reshape(size, size), 
                         cmap='RdBu', vmin=-1, vmax=1)
        axes[1, i].set_title(f'Retrieved')
        axes[1, i].axis('off')
        
        # Overlap score
        color = 'green' if result['retrieval_successful'] else 'red'
        axes[2, i].bar(0, result['overlap_with_original'], color=color, alpha=0.7)
        axes[2, i].set_ylim(0, 1)
        axes[2, i].set_title(f'Overlap: {result["overlap_with_original"]:.2f}')
        axes[2, i].set_xticks([])
        axes[2, i].axhline(y=0.8, color='black', linestyle='--', alpha=0.5)
    
    # Add row labels
    axes[0, 0].set_ylabel('Input Pattern', fontsize=12)
    axes[1, 0].set_ylabel('Retrieved Pattern', fontsize=12)
    axes[2, 0].set_ylabel('Overlap Score', fontsize=12)
    
    plt.tight_layout()
    plot_path = Path(PLOTS_DIR) / 'spatial_invariance_limitation.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = ['green' if overlap > 0.8 else 'red' for overlap in overlaps]
    bars = ax.bar(range(len(labels)), overlaps, color=colors, alpha=0.7)
    
    ax.set_xlabel('Shift Type')
    ax.set_ylabel('Overlap with Original Pattern')
    ax.set_title('Spatial Invariance Test Results\n(Red = Failed Retrieval, Green = Successful)')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.axhline(y=0.8, color='black', linestyle='--', alpha=0.5, label='Success Threshold')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_path = Path(PLOTS_DIR) / 'spatial_invariance_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Spatial invariance plots saved")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def close_all_figures() -> None:
    """Close all matplotlib figures to prevent memory leaks."""
    plt.close('all')
    logger.debug("All matplotlib figures closed")


def set_visualization_style(style: str = 'seaborn-v0_8') -> None:
    """
    Set consistent visualization style for all plots.
    
    Args:
        style: Matplotlib style to use
    """
    try:
        plt.style.use(style)
        logger.info(f"Visualization style set to: {style}")
    except OSError:
        logger.warning(f"Style '{style}' not available, using default")


def save_all_open_figures(directory: str, prefix: str = "figure") -> List[str]:
    """
    Save all currently open figures to a directory.
    
    Args:
        directory: Directory to save figures
        prefix: Prefix for filenames
        
    Returns:
        List of saved file paths
    """
    import os
    os.makedirs(directory, exist_ok=True)
    
    saved_paths = []
    for i in plt.get_fignums():
        fig = plt.figure(i)
        filepath = os.path.join(directory, f"{prefix}_{i}.png")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        saved_paths.append(filepath)
    
    logger.info(f"Saved {len(saved_paths)} figures to {directory}")
    return saved_paths


if __name__ == "__main__":
    # Demonstration of visualization capabilities
    print("Hopfield Network Visualization Module")
    print("=====================================")
    
    # Create sample pattern for testing
    test_pattern = np.random.choice([PATTERN_OFF, PATTERN_ON], size=100)
    
    # Test console display
    print("\\n1. Console Pattern Display:")
    display_pattern(test_pattern, "Sample Pattern")
    
    # Test matplotlib visualization
    print("\\n2. Matplotlib Pattern Visualization:")
    fig = visualize_pattern(test_pattern, "Sample Pattern Visualization", show=False)
    plt.close(fig)
    
    print("\\nVisualization module ready for use!")
    print("All plotting functions consolidated in visualize.py")