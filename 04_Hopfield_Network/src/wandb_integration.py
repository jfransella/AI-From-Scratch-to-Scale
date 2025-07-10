"""
Weights & Biases Integration for Hopfield Network Visualization
=============================================================

This module provides seamless integration between the Hopfield Network implementation
and Weights & Biases experiment tracking platform, following the established patterns
from other models in the AI-From-Scratch-to-Scale project.

Educational Objectives:
- Systematic experiment tracking and comparison
- Interactive visualizations and dashboards
- Reproducible research practices
- Professional ML development workflow
- Enhanced collaboration and sharing capabilities

Key Features:
- Optional W&B integration (can be disabled with --no-wandb)
- Automatic logging of network configurations and hyperparameters
- Interactive plots for capacity analysis, energy landscapes, and convergence
- Comprehensive experiment comparison across different network sizes
- Educational dashboards for understanding Hopfield network behavior
"""

import logging
import os
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from .config import (
    WANDB_PROJECT_NAME, WANDB_ENTITY, LOG_VISUALIZATIONS, 
    SAVE_MODEL_ARTIFACTS, PLOTS_DIR
)

logger = logging.getLogger(__name__)


class WandbVisualizer:
    """
    Weights & Biases integration for Hopfield Network experiments.
    
    This class provides a seamless interface for logging experiments, metrics,
    and visualizations to Weights & Biases while maintaining compatibility
    with local-only execution when W&B is disabled.
    
    Educational Focus:
    - Demonstrates professional ML experiment tracking
    - Shows systematic hyperparameter exploration
    - Enables easy comparison across experiments
    - Provides interactive dashboards for learning
    """
    
    def __init__(self, wandb_run: Optional[Any] = None, enabled: bool = True) -> None:
        """
        Initialize the W&B visualizer.
        
        Args:
            wandb_run: Active Weights & Biases run object
            enabled: Whether to enable W&B logging
            
        Raises:
            ImportError: If wandb is not installed but enabled=True
            ValueError: If wandb_run is None when enabled=True
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.wandb_run = wandb_run
        
        if enabled and not WANDB_AVAILABLE:
            logger.warning(
                "Weights & Biases not available. Install with: pip install wandb\n"
                "Continuing with local logging only..."
            )
            self.enabled = False
        
        if self.enabled and wandb_run is None:
            raise ValueError(
                "wandb_run cannot be None when visualization is enabled. "
                "Initialize wandb.init() first or set enabled=False"
            )
        
        if self.enabled:
            logger.info("W&B visualizer initialized - experiment tracking enabled")
        else:
            logger.info("W&B visualizer initialized - local logging only")
    
    def log_network_config(self, network_size: int, stored_patterns: int, 
                          theoretical_capacity: int, config: Dict[str, Any]) -> None:
        """
        Log network configuration and hyperparameters.
        
        Args:
            network_size: Number of neurons in the network
            stored_patterns: Number of patterns stored
            theoretical_capacity: Theoretical storage capacity
            config: Additional configuration parameters
        """
        metrics = {
            "network/size": network_size,
            "network/stored_patterns": stored_patterns,
            "network/theoretical_capacity": theoretical_capacity,
            "network/capacity_utilization": stored_patterns / theoretical_capacity,
            "network/memory_usage_mb": (network_size ** 2 * 8) / (1024 * 1024),  # float64
        }
        
        # Add config parameters
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool)):
                metrics[f"config/{key}"] = value
        
        self._log_metrics(metrics)
        logger.info(f"Network configuration logged: {network_size} neurons, {stored_patterns} patterns")
    
    def log_experiment_results(self, experiment_name: str, results: Dict[str, Any], 
                             step: Optional[int] = None) -> None:
        """
        Log experiment results and performance metrics.
        
        Args:
            experiment_name: Name of the experiment
            results: Dictionary containing experiment results
            step: Optional step number for time series logging
        """
        metrics = {}
        
        # Process results dictionary
        for key, value in results.items():
            if isinstance(value, (int, float)):
                metrics[f"{experiment_name}/{key}"] = value
            elif isinstance(value, dict):
                # Flatten nested dictionaries
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        metrics[f"{experiment_name}/{key}_{subkey}"] = subvalue
            elif isinstance(value, (list, np.ndarray)):
                # Log statistical summaries for arrays
                if len(value) > 0:
                    try:
                        # Try to convert to numpy array
                        array_vals = np.array(value)
                        # Check if array is homogeneous (no nested structures)
                        if array_vals.dtype == 'object':
                            # Handle nested arrays (like energy_histories)
                            if key == 'energy_histories' and isinstance(value[0], list):
                                # For energy histories, log statistics about the histories themselves
                                history_lengths = [len(hist) for hist in value]
                                metrics[f"{experiment_name}/{key}_count"] = len(value)
                                metrics[f"{experiment_name}/{key}_avg_length"] = np.mean(history_lengths)
                                metrics[f"{experiment_name}/{key}_max_length"] = np.max(history_lengths)
                                metrics[f"{experiment_name}/{key}_min_length"] = np.min(history_lengths)
                            else:
                                # For other object arrays, just log the count
                                metrics[f"{experiment_name}/{key}_count"] = len(value)
                        else:
                            # Regular homogeneous array - log statistics
                            metrics[f"{experiment_name}/{key}_mean"] = np.mean(array_vals)
                            metrics[f"{experiment_name}/{key}_std"] = np.std(array_vals)
                            metrics[f"{experiment_name}/{key}_min"] = np.min(array_vals)
                            metrics[f"{experiment_name}/{key}_max"] = np.max(array_vals)
                    except (ValueError, TypeError) as e:
                        # If conversion fails, just log the count
                        metrics[f"{experiment_name}/{key}_count"] = len(value)
        
        self._log_metrics(metrics, step=step)
        logger.info(f"Experiment results logged: {experiment_name}")
    
    def log_capacity_analysis(self, results: Dict[int, Dict[str, float]], 
                            network_sizes: List[int], pattern_counts: List[int], 
                            success_rates: List[float]) -> None:
        """
        Log capacity analysis results for systematic study.
        
        Args:
            results: Dictionary of results from capacity experiment
            network_sizes: List of network sizes tested
            pattern_counts: List of pattern counts for each test
            success_rates: List of success rates achieved
        """
        # Create capacity analysis table
        if self.enabled and LOG_VISUALIZATIONS:
            table_data = []
            for size, patterns, success in zip(network_sizes, pattern_counts, success_rates):
                theoretical_capacity = int(0.15 * size)
                capacity_ratio = patterns / theoretical_capacity
                table_data.append([size, patterns, theoretical_capacity, 
                                 capacity_ratio, success])
            
            table = wandb.Table(
                columns=["Network Size", "Stored Patterns", "Theoretical Capacity", 
                        "Capacity Ratio", "Success Rate"],
                data=table_data
            )
            
            self.wandb_run.log({"capacity_analysis/results_table": table})
        
        # Log summary metrics
        if len(success_rates) > 0:
            metrics = {
                "capacity_analysis/mean_success_rate": np.mean(success_rates),
                "capacity_analysis/std_success_rate": np.std(success_rates),
                "capacity_analysis/max_network_size": max(network_sizes) if network_sizes else 0,
                "capacity_analysis/total_experiments": len(success_rates)
            }
            self._log_metrics(metrics)
    
    def log_figure(self, figure: matplotlib.figure.Figure, name: str, 
                  step: Optional[int] = None, close_figure: bool = True) -> None:
        """
        Log a matplotlib figure to W&B and save locally.
        
        Args:
            figure: Matplotlib figure to log
            name: Name for the figure
            step: Optional step number
            close_figure: Whether to close the figure after logging
        """
        # Save locally
        local_path = os.path.join(PLOTS_DIR, f"{name}.png")
        figure.savefig(local_path, dpi=300, bbox_inches='tight')
        
        # Log to W&B
        if self.enabled and LOG_VISUALIZATIONS:
            self.wandb_run.log({f"plots/{name}": wandb.Image(figure)}, step=step)
        
        if close_figure:
            plt.close(figure)
        
        logger.info(f"Figure logged: {name}")
    
    def log_energy_landscape(self, energy_values: np.ndarray, state_labels: List[str],
                           step: Optional[int] = None) -> None:
        """
        Log energy landscape visualization.
        
        Args:
            energy_values: Array of energy values
            state_labels: Labels for different states
            step: Optional step number
        """
        if self.enabled and LOG_VISUALIZATIONS:
            # Create energy landscape plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot energy distribution
            ax.hist(energy_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Energy')
            ax.set_ylabel('Frequency')
            ax.set_title('Energy Landscape Distribution')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            ax.axvline(np.mean(energy_values), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(energy_values):.2f}')
            ax.axvline(np.median(energy_values), color='orange', linestyle='--', 
                      label=f'Median: {np.median(energy_values):.2f}')
            ax.legend()
            
            self.log_figure(fig, "energy_landscape", step=step)
        
        # Log energy statistics
        metrics = {
            "energy/mean": np.mean(energy_values),
            "energy/std": np.std(energy_values),
            "energy/min": np.min(energy_values),
            "energy/max": np.max(energy_values),
            "energy/range": np.max(energy_values) - np.min(energy_values)
        }
        self._log_metrics(metrics, step=step)
    
    def log_convergence_analysis(self, convergence_steps: List[int], 
                               experiment_name: str, step: Optional[int] = None) -> None:
        """
        Log convergence analysis results.
        
        Args:
            convergence_steps: List of steps required for convergence
            experiment_name: Name of the experiment
            step: Optional step number
        """
        if len(convergence_steps) == 0:
            return
        
        steps_array = np.array(convergence_steps)
        
        # Log convergence statistics
        metrics = {
            f"convergence/{experiment_name}_mean_steps": np.mean(steps_array),
            f"convergence/{experiment_name}_std_steps": np.std(steps_array),
            f"convergence/{experiment_name}_max_steps": np.max(steps_array),
            f"convergence/{experiment_name}_min_steps": np.min(steps_array),
            f"convergence/{experiment_name}_fast_convergence_rate": np.mean(steps_array <= 3)
        }
        self._log_metrics(metrics, step=step)
        
        # Create convergence histogram
        if self.enabled and LOG_VISUALIZATIONS:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(steps_array, bins=range(1, max(steps_array) + 2), 
                   alpha=0.7, color='lightgreen', edgecolor='black')
            ax.set_xlabel('Convergence Steps')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Convergence Steps Distribution - {experiment_name}')
            ax.grid(True, alpha=0.3)
            
            # Add mean line
            ax.axvline(np.mean(steps_array), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(steps_array):.1f}')
            ax.legend()
            
            self.log_figure(fig, f"convergence_{experiment_name}", step=step)
    
    def log_pattern_analysis(self, patterns: np.ndarray, labels: List[str],
                           experiment_name: str, step: Optional[int] = None) -> None:
        """
        Log pattern visualization and analysis.
        
        Args:
            patterns: Array of patterns to visualize
            labels: Labels for each pattern
            experiment_name: Name of the experiment
            step: Optional step number
        """
        if not self.enabled or not LOG_VISUALIZATIONS:
            return
        
        # Create pattern grid visualization
        n_patterns = min(len(patterns), 20)  # Limit to 20 patterns for readability
        grid_size = int(np.sqrt(patterns.shape[1]))
        
        if grid_size * grid_size == patterns.shape[1]:  # Square patterns
            cols = min(5, n_patterns)
            rows = (n_patterns + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            if rows == 1:
                axes = axes.reshape(1, -1)
            if cols == 1:
                axes = axes.reshape(-1, 1)
            
            for i in range(n_patterns):
                row, col = divmod(i, cols)
                pattern_2d = patterns[i].reshape(grid_size, grid_size)
                
                im = axes[row, col].imshow(pattern_2d, cmap='RdBu', vmin=-1, vmax=1)
                axes[row, col].set_title(f'{labels[i] if i < len(labels) else f"Pattern {i}"}')
                axes[row, col].axis('off')
            
            # Hide empty subplots
            for i in range(n_patterns, rows * cols):
                row, col = divmod(i, cols)
                axes[row, col].axis('off')
            
            plt.tight_layout()
            self.log_figure(fig, f"patterns_{experiment_name}", step=step)
    
    def create_experiment_summary(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Create a comprehensive experiment summary table.
        
        Args:
            all_results: Dictionary of all experiment results
        """
        if not self.enabled or not LOG_VISUALIZATIONS:
            return
        
        # Create summary table
        table_data = []
        for exp_name, results in all_results.items():
            row = [exp_name]
            
            # Extract key metrics
            if 'success_rate' in results:
                row.append(f"{results['success_rate']:.2%}")
            else:
                row.append("N/A")
            
            if 'mean_convergence_steps' in results:
                row.append(f"{results['mean_convergence_steps']:.1f}")
            else:
                row.append("N/A")
            
            if 'total_tests' in results:
                row.append(results['total_tests'])
            else:
                row.append("N/A")
            
            table_data.append(row)
        
        table = wandb.Table(
            columns=["Experiment", "Success Rate", "Avg. Convergence", "Total Tests"],
            data=table_data
        )
        
        self.wandb_run.log({"experiment_summary": table})
        logger.info("Experiment summary table created")
    
    def save_model_artifact(self, model_state: Dict[str, Any], artifact_name: str) -> None:
        """
        Save model state as W&B artifact.
        
        Args:
            model_state: Dictionary containing model state
            artifact_name: Name for the artifact
        """
        if not self.enabled or not SAVE_MODEL_ARTIFACTS:
            return
        
        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description="Hopfield Network model state and configuration"
        )
        
        # Save model state to temporary file
        temp_path = os.path.join(PLOTS_DIR, f"{artifact_name}.npz")
        np.savez(temp_path, **model_state)
        
        # Add file to artifact
        artifact.add_file(temp_path)
        
        # Log artifact
        self.wandb_run.log_artifact(artifact)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        logger.info(f"Model artifact saved: {artifact_name}")
    
    def log_file_artifact(self, file_path: str, artifact_name: str, description: str = "") -> None:
        """
        Log a file as an artifact to W&B.
        
        Args:
            file_path: Path to the file to log
            artifact_name: Name for the artifact
            description: Optional description of the artifact
        """
        if not self.enabled:
            logger.info(f"File artifact logged locally: {artifact_name} at {file_path}")
            return
        
        try:
            import wandb
            
            # Create artifact
            artifact = wandb.Artifact(
                name=artifact_name,
                type="dataset" if "report" in artifact_name else "model",
                description=description
            )
            
            # Add file to artifact
            artifact.add_file(file_path)
            
            # Log artifact
            wandb.log_artifact(artifact)
            
            logger.info(f"File artifact logged to W&B: {artifact_name}")
            
        except Exception as e:
            logger.warning(f"Failed to log file artifact to W&B: {e}")
    
    def log_noise_robustness(self, results: Dict[float, Dict[str, float]], 
                           pattern_type: str = "patterns") -> None:
        """
        Log noise robustness experiment results with interactive visualizations.
        
        Args:
            results: Dictionary with noise levels as keys and metrics as values
                    Each value should contain 'success_rate', 'success_rate_std',
                    'avg_overlap_improvement', 'overlap_improvement_std'
            pattern_type: Type of patterns used in the experiment
            
        Educational Focus:
        - Demonstrates associative memory's error correction capabilities
        - Shows robustness to partial or corrupted information
        - Visualizes the relationship between noise level and retrieval quality
        """
        if not self.enabled:
            return
            
        try:
            # Extract data for plotting
            noise_levels = sorted(list(results.keys()))
            success_rates = [results[n]['success_rate'] for n in noise_levels]
            success_errors = [results[n]['success_rate_std'] for n in noise_levels]
            overlap_improvements = [results[n]['avg_overlap_improvement'] for n in noise_levels]
            overlap_errors = [results[n]['overlap_improvement_std'] for n in noise_levels]
            
            # Create comprehensive visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Success rate vs noise level
            ax1.errorbar(noise_levels, success_rates, yerr=success_errors,
                        marker='o', linewidth=2, capsize=5, color='blue', 
                        markersize=8, label='Success Rate')
            ax1.set_xlabel('Noise Level (Fraction of Bits Flipped)')
            ax1.set_ylabel('Retrieval Success Rate')
            ax1.set_title(f'Noise Robustness - Success Rate ({pattern_type})')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_ylim(0, 1.1)
            
            # Plot 2: Overlap improvement vs noise level
            ax2.errorbar(noise_levels, overlap_improvements, yerr=overlap_errors,
                        marker='s', linewidth=2, capsize=5, color='green',
                        markersize=8, label='Overlap Improvement')
            ax2.set_xlabel('Noise Level (Fraction of Bits Flipped)')
            ax2.set_ylabel('Average Overlap Improvement')
            ax2.set_title('Pattern Recovery Quality')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Plot 3: Combined success rate and improvement
            ax3_twin = ax3.twinx()
            line1 = ax3.plot(noise_levels, success_rates, 'o-', color='blue', 
                           linewidth=2, markersize=6, label='Success Rate')
            line2 = ax3_twin.plot(noise_levels, overlap_improvements, 's-', color='green',
                                linewidth=2, markersize=6, label='Overlap Improvement')
            
            ax3.set_xlabel('Noise Level')
            ax3.set_ylabel('Success Rate', color='blue')
            ax3_twin.set_ylabel('Overlap Improvement', color='green')
            ax3.set_title('Combined Performance Metrics')
            ax3.grid(True, alpha=0.3)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='center right')
            
            # Plot 4: Error magnitude visualization
            total_error = np.array(success_errors) + np.array(overlap_errors)
            ax4.bar(range(len(noise_levels)), total_error, 
                   color='orange', alpha=0.7, label='Combined Error')
            ax4.set_xlabel('Noise Level Index')
            ax4.set_ylabel('Total Standard Deviation')
            ax4.set_title('Measurement Uncertainty')
            ax4.set_xticks(range(len(noise_levels)))
            ax4.set_xticklabels([f'{n:.2f}' for n in noise_levels])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure for logging
            noise_plot_path = os.path.join(PLOTS_DIR, f'noise_robustness_{pattern_type}_detailed.png')
            os.makedirs(PLOTS_DIR, exist_ok=True)
            plt.savefig(noise_plot_path, dpi=300, bbox_inches='tight')
            
            # Log the figure to W&B
            self.log_figure(fig, f"noise_robustness/performance_analysis_{pattern_type}")
            
            # Also log as image
            self.log_image(noise_plot_path, f"noise_robustness/detailed_plot_{pattern_type}",
                          f"Comprehensive noise robustness analysis for {pattern_type}")
            
            # Log summary metrics table
            noise_summary = {}
            for i, noise_level in enumerate(noise_levels):
                noise_summary[f"noise_{noise_level:.2f}_success_rate"] = success_rates[i]
                noise_summary[f"noise_{noise_level:.2f}_overlap_improvement"] = overlap_improvements[i]
                noise_summary[f"noise_{noise_level:.2f}_combined_performance"] = (
                    success_rates[i] * 0.7 + overlap_improvements[i] * 0.3
                )
            
            # Log to W&B table for detailed analysis
            self.wandb_run.log({
                "noise_robustness_summary": wandb.Table(
                    columns=["noise_level", "success_rate", "success_std", 
                            "overlap_improvement", "overlap_std", "combined_score"],
                    data=[[f"{n:.2f}", 
                          f"{results[n]['success_rate']:.4f}",
                          f"{results[n]['success_rate_std']:.4f}",
                          f"{results[n]['avg_overlap_improvement']:.4f}",
                          f"{results[n]['overlap_improvement_std']:.4f}",
                          f"{results[n]['success_rate'] * 0.7 + results[n]['avg_overlap_improvement'] * 0.3:.4f}"]
                         for n in noise_levels]
                )
            })
            
            plt.close(fig)
            logger.info(f"Noise robustness analysis logged to W&B for {pattern_type}")
            
        except Exception as e:
            logger.warning(f"Failed to log noise robustness analysis to W&B: {e}")
    
    def _log_metrics(self, metrics: Dict[str, Union[int, float]], 
                    step: Optional[int] = None) -> None:
        """
        Internal method to log metrics to W&B.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if self.enabled:
            self.wandb_run.log(metrics, step=step)
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], 
                    step: Optional[int] = None) -> None:
        """
        Log metrics to W&B and local storage.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for time series logging
        """
        self._log_metrics(metrics, step)
    
    def log_image(self, image_path: str, key: str, caption: str = "") -> None:
        """
        Log an image to W&B.
        
        Args:
            image_path: Path to the image file
            key: W&B logging key
            caption: Optional caption for the image
        """
        if not self.enabled:
            logger.debug(f"Image logging disabled: {key}")
            return
        
        try:
            import wandb
            from PIL import Image
            
            # Load and log the image
            image = Image.open(image_path)
            wandb.log({key: wandb.Image(image, caption=caption)})
            logger.info(f"Image logged to W&B: {key}")
            
        except Exception as e:
            logger.warning(f"Failed to log image {key}: {e}")

    def create_comprehensive_comparison(self, experiment_results: Dict[str, Any]) -> None:
        """
        Create comprehensive comparison visualization across all experiments.
        
        Args:
            experiment_results: Dictionary containing results from all experiments
        """
        if not self.enabled:
            logger.debug("W&B not available - skipping comprehensive comparison")
            return
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create a comprehensive comparison figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Hopfield Network - Comprehensive Experiment Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Capacity Analysis
            if 'capacity' in experiment_results:
                capacity_data = experiment_results['capacity']
                network_sizes = []
                success_rates = []
                
                for key, value in capacity_data.items():
                    if key.endswith('_success_rate'):
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
                    axes[0, 0].text(0.5, 0.5, 'No capacity data available', ha='center', va='center', transform=axes[0, 0].transAxes)
            else:
                axes[0, 0].text(0.5, 0.5, 'No capacity data available', ha='center', va='center', transform=axes[0, 0].transAxes)
            
            # Plot 2: Noise Robustness
            if 'noise_robustness' in experiment_results:
                noise_data = experiment_results.get('noise_robustness', {})
                noise_levels = []
                success_rates = []
                
                for key, value in noise_data.items():
                    if 'success_rate_' in key and not key.endswith('_std'):
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
                    axes[0, 1].text(0.5, 0.5, 'No noise data available', ha='center', va='center', transform=axes[0, 1].transAxes)
            else:
                axes[0, 1].text(0.5, 0.5, 'No noise data available', ha='center', va='center', transform=axes[0, 1].transAxes)
            
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
                axes[1, 0].text(0.5, 0.5, 'No convergence data available', ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # Plot 4: Overall Performance Summary
            experiments = []
            performance_scores = []
            
            if 'basic_training' in experiment_results:
                experiments.append('Basic\nTraining')
                basic_score = experiment_results['basic_training'].get('perfect_recall_rate', 0)
                performance_scores.append(basic_score)
            
            if 'capacity' in experiment_results:
                experiments.append('Capacity\nAnalysis')
                capacity_scores = [v for k, v in experiment_results['capacity'].items() 
                                 if k.endswith('_success_rate')]
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
                axes[1, 1].text(0.5, 0.5, 'No performance data available', ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.tight_layout()
            
            # Log to W&B
            self.log_figure(fig, "comprehensive_comparison", close_figure=True)
            logger.info("Comprehensive comparison visualization created and logged")
            
        except Exception as e:
            logger.warning(f"Failed to create comprehensive comparison: {e}")


def initialize_wandb(project_name: str = WANDB_PROJECT_NAME,
                    entity: Optional[str] = WANDB_ENTITY,
                    config: Optional[Dict[str, Any]] = None,
                    enabled: bool = True) -> Tuple[Any, WandbVisualizer]:
    """
    Initialize Weights & Biases run and visualizer.
    
    Args:
        project_name: W&B project name
        entity: W&B entity (username or team)
        config: Configuration dictionary to log
        enabled: Whether to enable W&B logging
    
    Returns:
        Tuple of (wandb_run, visualizer)
    """
    if not enabled or not WANDB_AVAILABLE:
        logger.info("W&B integration disabled")
        return None, WandbVisualizer(enabled=False)
    
    # Initialize W&B run
    wandb_run = wandb.init(
        project=project_name,
        entity=entity,
        config=config or {},
        mode="online" if enabled else "disabled"
    )
    
    # Create visualizer
    visualizer = WandbVisualizer(wandb_run, enabled=True)
    
    logger.info(f"W&B run initialized: {wandb_run.name}")
    return wandb_run, visualizer


def finish_wandb(wandb_run: Optional[Any]) -> None:
    """
    Finish W&B run gracefully.
    
    Args:
        wandb_run: W&B run object to finish
    """
    if wandb_run is not None:
        wandb_run.finish()
        logger.info("W&B run finished successfully")
