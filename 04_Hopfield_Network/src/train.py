"""
Training Script for Hopfield Network: Energy-Based Pattern Storage
================================================================

This module demonstrates the Hebbian learning process for storing patterns
in a Hopfield Network, contrasting it with gradient-based supervised learning.

Key Educational Concepts:
- Hebbian learning: "Neurons that fire together, wire together"
- No error signal required (unsupervised vs. supervised learning)
- One-shot learning vs. iterative optimization
- Memory capacity and interference effects
- Statistical mechanics approach to neural computation

Historical Context:
Hopfield (1982) showed how simple local learning rules could create
global computational properties like associative memory.

Example usage:
    # Run with W&B tracking (default)
    python -m src.train
    
    # Run without W&B tracking
    python -m src.train --no-wandb
    
    # Run interactive console demonstration
    python -m src.train --experiment demo --no-wandb
    
    # Run MNIST demonstration  
    python -m src.train --experiment mnist
    
    # Run spatial invariance demonstration
    python -m src.train --experiment spatial_invariance
    
    # Run specific experiment
    python -m src.train --experiment capacity_analysis
"""

import argparse
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import time

try:
    # Try relative imports first (when run as module)
    from .config import *
    from .data_loader import PatternGenerator, HopfieldDataLoader
    from .model import HopfieldNetwork
    from .wandb_integration import WandbVisualizer, initialize_wandb, finish_wandb
    from .visualize import (
        plot_capacity_results, plot_noise_robustness, 
        plot_convergence_statistics, create_comprehensive_comparison,
        visualize_energy_landscape, visualize_pattern_set,
        plot_spatial_invariance_results
    )
except ImportError:
    # Fall back to absolute imports (when run as script)
    from config import *
    from data_loader import PatternGenerator, HopfieldDataLoader
    from model import HopfieldNetwork
    from wandb_integration import WandbVisualizer, initialize_wandb, finish_wandb
    from visualize import (
        plot_capacity_results, plot_noise_robustness, 
        plot_convergence_statistics, create_comprehensive_comparison,
        visualize_energy_landscape, visualize_pattern_set
    )

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(LOGS_DIR) / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HopfieldTrainer:
    """
    Trainer class for Hopfield Networks focusing on educational insights.
    
    This class orchestrates various experiments to demonstrate the
    capabilities and limitations of energy-based associative memory.
    """
    
    def __init__(self, network_size: int = NETWORK_SIZE, 
                 wandb_visualizer: Optional[WandbVisualizer] = None):
        """
        Initialize the trainer.
        
        Args:
            network_size: Size of the Hopfield network
            wandb_visualizer: Optional W&B visualizer for experiment tracking
        """
        self.network_size = network_size
        self.network = HopfieldNetwork(network_size)
        
        # Calculate pattern dimensions for the given network size
        # Find the best rectangular dimensions that exactly match network size
        def find_best_dimensions(target_size):
            """Find height and width that multiply to exactly target_size."""
            sqrt_size = int(np.sqrt(target_size))
            
            # Try to find factors close to square
            for h in range(sqrt_size, 0, -1):
                if target_size % h == 0:
                    w = target_size // h
                    return h, w
            
            # Fallback: just use 1 x target_size
            return 1, target_size
        
        pattern_height, pattern_width = find_best_dimensions(network_size)
        
        self.data_loader = HopfieldDataLoader()
        # Update the pattern generator to match network size exactly
        self.data_loader.generator = PatternGenerator(height=pattern_height, width=pattern_width)
        
        self.experiment_results = {}
        self.visualizer = wandb_visualizer
        
        # Create output directories
        for directory in [OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        if USE_FIXED_SEED:
            np.random.seed(RANDOM_SEED)
        
        logger.info(f"HopfieldTrainer initialized with network size {network_size}")
        
        # Log network configuration to W&B
        if self.visualizer:
            theoretical_capacity = int(0.15 * network_size)
            config = {
                "network_size": network_size,
                "max_patterns": MAX_PATTERNS,
                "random_seed": RANDOM_SEED if USE_FIXED_SEED else None
            }
            self.visualizer.log_network_config(
                network_size, 0, theoretical_capacity, config
            )
    
    def train_basic_patterns(self, pattern_type: str = "simple_shapes") -> Dict[str, float]:
        """
        Train the network on basic patterns and evaluate storage.
        
        Args:
            pattern_type: Type of patterns to use ('simple_shapes', 'letters', 'random')
            
        Returns:
            Dictionary with training results
            
        Educational Focus:
            Demonstrates the basic Hebbian learning process and immediate
            storage without iterative training (contrast with backpropagation).
        """
        logger.info(f"Starting basic pattern training with {pattern_type}")
        
        # Load patterns
        patterns = self.data_loader.load_pattern_set(pattern_type)
        pattern_list = list(patterns.values())
        
        # Limit to maximum patterns for this experiment
        if len(pattern_list) > MAX_PATTERNS:
            pattern_list = pattern_list[:MAX_PATTERNS]
            logger.info(f"Limited to {MAX_PATTERNS} patterns for capacity reasons")
        
        # Store patterns (one-shot learning!)
        start_time = time.time()
        self.network.store_patterns(pattern_list)
        training_time = time.time() - start_time
        
        logger.info(f"Pattern storage completed in {training_time:.4f} seconds")
        
        # Evaluate storage quality
        results = self._evaluate_pattern_storage(patterns, pattern_list)
        results['training_time'] = training_time
        results['pattern_type'] = pattern_type
        results['num_patterns'] = len(pattern_list)
        
        # Save results
        self.experiment_results['basic_training'] = results
        
        # Log results to W&B
        if self.visualizer:
            self.visualizer.log_metrics({
                "basic_training/perfect_recall_rate": results['perfect_recall_rate'],
                "basic_training/pattern_overlap": results['pattern_overlap'],
                "basic_training/training_time": results['training_time'],
                "basic_training/num_patterns": results['num_patterns']
            })
            
            # Log pattern visualization using centralized function
            pattern_plot_path = Path(PLOTS_DIR) / f"stored_patterns_{pattern_type}.png"
            visualize_pattern_set(
                patterns, 
                f"Stored {pattern_type} Patterns",
                save_path=str(pattern_plot_path),
                show=True
            )
            self.visualizer.log_image(str(pattern_plot_path), f"patterns/{pattern_type}_stored")
        else:
            # Visualize patterns without W&B logging using centralized function
            visualize_pattern_set(
                patterns, 
                f"Stored {pattern_type} Patterns",
                save_path=str(Path(PLOTS_DIR) / f"stored_patterns_{pattern_type}.png"),
                show=True
            )
        
        logger.info(f"Basic training results: {results}")
        return results
    
    def _evaluate_pattern_storage(self, original_patterns: Dict[str, np.ndarray], 
                                 stored_patterns: List[np.ndarray]) -> Dict[str, float]:
        """
        Evaluate how well patterns were stored.
        
        Args:
            original_patterns: Original pattern dictionary
            stored_patterns: List of patterns that were stored
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Test perfect recall (no noise)
        perfect_recalls = 0
        total_energy_decrease = 0
        total_steps = 0
        
        for i, pattern in enumerate(stored_patterns):
            retrieved, info = self.network.retrieve_pattern(pattern)
            
            if info['successful_retrieval']:
                perfect_recalls += 1
            
            total_energy_decrease += info['energy_decrease']
            total_steps += info['converged_steps']
        
        # Calculate pattern statistics
        stats = self.data_loader.analyze_pattern_statistics(
            {f"pattern_{i}": p for i, p in enumerate(stored_patterns)}
        )
        
        results = {
            'perfect_recall_rate': perfect_recalls / len(stored_patterns),
            'avg_energy_decrease': total_energy_decrease / len(stored_patterns),
            'avg_convergence_steps': total_steps / len(stored_patterns),
            'pattern_overlap': stats['mean_abs_overlap'],
            'network_capacity_ratio': len(stored_patterns) / (0.15 * self.network_size)
        }
        
        return results
    
    def experiment_storage_capacity(self) -> Dict[int, Dict[str, float]]:
        """
        Systematic experiment on storage capacity.
        
        Returns:
            Results for different numbers of stored patterns
            
        Educational Objective:
            Demonstrate the theoretical capacity limit (~0.15 * N) and
            show how performance degrades beyond this limit.
        """
        logger.info("Starting storage capacity experiment")
        
        capacity_results = {}
        
        for num_patterns in CAPACITY_EXPERIMENT_SIZES:
            if num_patterns > self.network_size:
                continue  # Skip if more patterns than neurons
            
            logger.info(f"Testing capacity with {num_patterns} patterns")
            
            trial_results = []
            
            for trial in range(CAPACITY_TRIALS):
                # Generate random patterns for this trial (lower density for better separation)
                random_patterns = self.data_loader.generator.generate_random_patterns(num_patterns, density=0.2)
                
                # Reset and store patterns
                self.network = HopfieldNetwork(self.network_size)
                self.network.store_patterns(random_patterns)
                
                # Test retrieval with noise
                successful_retrievals = 0
                total_energy_decrease = 0
                
                for pattern in random_patterns:
                    # Add moderate noise
                    noisy_pattern = self.data_loader.generator.add_noise(pattern, 0.2)
                    
                    # Attempt retrieval
                    retrieved, info = self.network.retrieve_pattern(noisy_pattern)
                    
                    # Use capacity-specific threshold (60% for random patterns)
                    capacity_threshold = 0.6  # More realistic for random patterns
                    if info['best_overlap'] >= capacity_threshold:
                        successful_retrievals += 1
                    
                    total_energy_decrease += info['energy_decrease']
                
                trial_result = {
                    'success_rate': successful_retrievals / num_patterns,
                    'avg_energy_decrease': total_energy_decrease / num_patterns,
                    'theoretical_ratio': num_patterns / (0.15 * self.network_size)
                }
                
                trial_results.append(trial_result)
            
            # Average across trials
            capacity_results[num_patterns] = {
                'success_rate': np.mean([r['success_rate'] for r in trial_results]),
                'success_rate_std': np.std([r['success_rate'] for r in trial_results]),
                'avg_energy_decrease': np.mean([r['avg_energy_decrease'] for r in trial_results]),
                'theoretical_ratio': trial_results[0]['theoretical_ratio']
            }
            
            logger.info(f"Capacity {num_patterns}: Success rate {capacity_results[num_patterns]['success_rate']:.3f}")
        
        # Save and visualize results
        self.experiment_results['capacity'] = capacity_results
        
        # Log intermediate results to W&B
        if self.visualizer:
            for num_patterns, metrics in capacity_results.items():
                self.visualizer.log_metrics({
                    f"capacity_experiment/success_rate_{num_patterns}": metrics['success_rate'],
                    f"capacity_experiment/theoretical_ratio_{num_patterns}": metrics['theoretical_ratio'],
                }, step=num_patterns)
        
        plot_capacity_results(capacity_results, self.network_size, self.visualizer)
        
        return capacity_results
    
    def experiment_noise_robustness(self, pattern_type: str = "simple_shapes") -> Dict[float, Dict[str, float]]:
        """
        Experiment on robustness to noise.
        
        Args:
            pattern_type: Type of patterns to use
            
        Returns:
            Results for different noise levels
            
        Educational Objective:
            Show how associative memory can recover from partial information,
            demonstrating error correction capabilities.
        """
        logger.info(f"Starting noise robustness experiment with {pattern_type}")
        
        # Load and store patterns
        patterns = self.data_loader.load_pattern_set(pattern_type)
        pattern_list = list(patterns.values())[:5]  # Use subset for noise testing
        
        self.network = HopfieldNetwork(self.network_size)
        self.network.store_patterns(pattern_list)
        
        noise_results = {}
        
        for noise_level in NOISE_EXPERIMENT_LEVELS:
            logger.info(f"Testing noise level {noise_level:.2f}")
            
            trial_results = []
            
            for trial in range(NOISE_TRIALS):
                successful_retrievals = 0
                total_overlap_improvement = 0
                
                for pattern in pattern_list:
                    # Add noise
                    noisy_pattern = self.data_loader.generator.add_noise(pattern, noise_level)
                    initial_overlap = self.data_loader.generator.calculate_overlap(pattern, noisy_pattern)
                    
                    # Retrieve pattern
                    retrieved, info = self.network.retrieve_pattern(noisy_pattern)
                    final_overlap = self.data_loader.generator.calculate_overlap(pattern, retrieved)
                    
                    if info['successful_retrieval']:
                        successful_retrievals += 1
                    
                    overlap_improvement = final_overlap - initial_overlap
                    total_overlap_improvement += overlap_improvement
                
                trial_result = {
                    'success_rate': successful_retrievals / len(pattern_list),
                    'avg_overlap_improvement': total_overlap_improvement / len(pattern_list)
                }
                
                trial_results.append(trial_result)
            
            # Average across trials
            noise_results[noise_level] = {
                'success_rate': np.mean([r['success_rate'] for r in trial_results]),
                'success_rate_std': np.std([r['success_rate'] for r in trial_results]),
                'avg_overlap_improvement': np.mean([r['avg_overlap_improvement'] for r in trial_results]),
                'overlap_improvement_std': np.std([r['avg_overlap_improvement'] for r in trial_results])
            }
            
            logger.info(f"Noise {noise_level:.2f}: Success rate {noise_results[noise_level]['success_rate']:.3f}")
        
        # Save and visualize results
        self.experiment_results['noise_robustness'] = noise_results
        
        # Log intermediate results to W&B
        if self.visualizer:
            for noise_level, metrics in noise_results.items():
                self.visualizer.log_metrics({
                    f"noise_experiment/success_rate_{noise_level}": metrics['success_rate'],
                    f"noise_experiment/overlap_improvement_{noise_level}": metrics['avg_overlap_improvement'],
                })  # Remove step parameter to avoid monotonic step warnings
        
        plot_noise_robustness(noise_results, pattern_type, self.visualizer)
        
        return noise_results
    
    def experiment_convergence_dynamics(self) -> Dict[str, List[float]]:
        """
        Study convergence dynamics and energy landscape.
        
        Returns:
            Results on convergence behavior
            
        Educational Objective:
            Demonstrate Lyapunov stability and energy minimization,
            showing the physics-inspired approach to computation.
        """
        logger.info("Starting convergence dynamics experiment")
        
        # Create test patterns
        patterns = self.data_loader.load_pattern_set("simple_shapes")
        pattern_list = list(patterns.values())[:3]  # Use small set for detailed analysis
        
        self.network = HopfieldNetwork(self.network_size)
        self.network.store_patterns(pattern_list)
        
        convergence_results = {
            'energy_histories': [],
            'convergence_steps': [],
            'energy_decreases': [],
            'initial_overlaps': [],
            'final_overlaps': []
        }
        
        for trial in range(CONVERGENCE_TRIALS):
            # Select random pattern and add noise
            pattern_idx = np.random.choice(len(pattern_list))
            pattern = pattern_list[pattern_idx]
            noise_level = np.random.uniform(0.1, 0.4)
            noisy_pattern = self.data_loader.generator.add_noise(pattern, noise_level)
            
            # Calculate initial overlap
            initial_overlap = self.data_loader.generator.calculate_overlap(pattern, noisy_pattern)
            
            # Retrieve and analyze
            retrieved, info = self.network.retrieve_pattern(noisy_pattern)
            final_overlap = self.data_loader.generator.calculate_overlap(pattern, retrieved)
            
            # Store results
            convergence_results['energy_histories'].append(info['energy_history'])
            convergence_results['convergence_steps'].append(info['converged_steps'])
            convergence_results['energy_decreases'].append(info['energy_decrease'])
            convergence_results['initial_overlaps'].append(initial_overlap)
            convergence_results['final_overlaps'].append(final_overlap)
        
        # Save results
        self.experiment_results['convergence'] = convergence_results
        
        # Log convergence statistics to W&B
        if self.visualizer:
            avg_steps = np.mean(convergence_results['convergence_steps'])
            avg_energy_decrease = np.mean(convergence_results['energy_decreases'])
            avg_overlap_improvement = np.mean([
                f - i for i, f in zip(convergence_results['initial_overlaps'], convergence_results['final_overlaps'])
            ])
            
            self.visualizer.log_metrics({
                "convergence_experiment/avg_convergence_steps": avg_steps,
                "convergence_experiment/avg_energy_decrease": avg_energy_decrease,
                "convergence_experiment/avg_overlap_improvement": avg_overlap_improvement,
                "convergence_experiment/total_trials": len(convergence_results['convergence_steps'])
            })
        
        # Visualize convergence statistics
        self._plot_convergence_statistics(convergence_results)
        
        # Visualize energy landscape using centralized function
        energy_landscape_path = Path(PLOTS_DIR) / 'energy_landscape.png'
        visualize_energy_landscape(
            self.network.stored_patterns, 
            self.network.weights,
            save_path=str(energy_landscape_path),
            show=True
        )
        
        # Log energy landscape to W&B if available
        if self.visualizer:
            self.visualizer.log_image(str(energy_landscape_path), "convergence_analysis/energy_landscape")
        
        logger.info("Convergence dynamics experiment completed")
        return convergence_results
    
    def _plot_convergence_statistics(self, convergence_results: Dict[str, List[float]]) -> None:
        """
        Plot convergence dynamics statistics.
        
        Args:
            convergence_results: Results from convergence experiment
        """
        plot_path = Path(PLOTS_DIR) / 'convergence_statistics.png'
        
        # Use centralized visualization function
        fig = plot_convergence_statistics(
            convergence_results, 
            title="Convergence Dynamics Analysis",
            save_path=str(plot_path),
            show=True
        )
        
        # Log plot to W&B if available
        if self.visualizer and plot_path.exists():
            self.visualizer.log_image(str(plot_path), "convergence_analysis/statistics_plot")
        
        logger.info("Convergence statistics plot saved")
    
    def run_full_experimental_suite(self) -> Dict[str, Dict]:
        """
        Run the complete experimental suite for educational demonstration.
        
        Returns:
            Complete results from all experiments
        """
        logger.info("="*80)
        logger.info("STARTING FULL HOPFIELD NETWORK EXPERIMENTAL SUITE")
        logger.info("="*80)
        
        # Experiment 1: Basic pattern training
        logger.info("\n1. Basic Pattern Training")
        basic_results = self.train_basic_patterns("simple_shapes")
        
        # Experiment 2: Storage capacity
        logger.info("\n2. Storage Capacity Analysis")
        capacity_results = self.experiment_storage_capacity()
        
        # Experiment 3: Noise robustness
        logger.info("\n3. Noise Robustness Testing")
        noise_results = self.experiment_noise_robustness("simple_shapes")
        
        # Experiment 4: Convergence dynamics
        logger.info("\n4. Convergence Dynamics Study")
        convergence_results = self.experiment_convergence_dynamics()
        
        # Generate comprehensive report
        self._generate_experimental_report()
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENTAL SUITE COMPLETED")
        logger.info("="*80)
        
        return self.experiment_results
    
    def _generate_experimental_report(self) -> None:
        """
        Generate a comprehensive experimental report.
        """
        report_path = Path(OUTPUT_DIR) / 'hopfield_experimental_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("HOPFIELD NETWORK EXPERIMENTAL REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("NETWORK CONFIGURATION:\n")
            f.write(f"  Network Size: {self.network_size} neurons\n")
            f.write(f"  Theoretical Capacity: ~{int(0.15 * self.network_size)} patterns\n\n")
            
            # Basic training results
            if 'basic_training' in self.experiment_results:
                basic = self.experiment_results['basic_training']
                f.write("BASIC PATTERN TRAINING:\n")
                f.write(f"  Pattern Type: {basic['pattern_type']}\n")
                f.write(f"  Patterns Stored: {basic['num_patterns']}\n")
                f.write(f"  Perfect Recall Rate: {basic['perfect_recall_rate']:.3f}\n")
                f.write(f"  Training Time: {basic['training_time']:.4f} seconds\n\n")
            
            # Capacity results
            if 'capacity' in self.experiment_results:
                f.write("STORAGE CAPACITY ANALYSIS:\n")
                capacity = self.experiment_results['capacity']
                for num_patterns, results in capacity.items():
                    f.write(f"  {num_patterns} patterns: {results['success_rate']:.3f} ± {results['success_rate_std']:.3f}\n")
                f.write("\n")
            
            # Noise robustness results
            if 'noise_robustness' in self.experiment_results:
                f.write("NOISE ROBUSTNESS ANALYSIS:\n")
                noise = self.experiment_results['noise_robustness']
                for noise_level, results in noise.items():
                    f.write(f"  {noise_level:.1f} noise: {results['success_rate']:.3f} success rate\n")
                f.write("\n")
            
            f.write("EDUCATIONAL INSIGHTS:\n")
            f.write("  1. Hopfield networks demonstrate energy-based learning\n")
            f.write("  2. Hebbian learning enables one-shot pattern storage\n")
            f.write("  3. Network capacity is limited by pattern interference\n")
            f.write("  4. Energy minimization guarantees convergence\n")
            f.write("  5. Associative memory enables error correction\n\n")
            
        logger.info(f"Experimental report saved to {report_path}")
    
    def run_capacity_experiment(self) -> Dict[int, Dict[str, float]]:
        """Wrapper for capacity experiment with W&B logging."""
        results = self.experiment_storage_capacity()
        
        # Log results to W&B
        if self.visualizer:
            # Log capacity metrics
            for num_patterns, metrics in results.items():
                self.visualizer.log_metrics({
                    f"capacity/success_rate_{num_patterns}": metrics['success_rate'],
                    f"capacity/theoretical_ratio_{num_patterns}": metrics['theoretical_ratio'],
                    f"capacity/energy_decrease_{num_patterns}": metrics['avg_energy_decrease']
                })
            
            # Create and log capacity plot
            pattern_counts = list(results.keys())
            success_rates = [results[n]['success_rate'] for n in pattern_counts]
            network_sizes = [self.network_size] * len(pattern_counts)  # Same network size for all tests
            self.visualizer.log_capacity_analysis(results, network_sizes, pattern_counts, success_rates)
        
        return results
    
    def run_noise_robustness_experiment(self, pattern_type: str = "simple_shapes") -> Dict[float, Dict[str, float]]:
        """Wrapper for noise robustness experiment with W&B logging."""
        results = self.experiment_noise_robustness(pattern_type)
        
        # Log results to W&B
        if self.visualizer:
            # Log noise robustness metrics
            for noise_level, metrics in results.items():
                self.visualizer.log_metrics({
                    f"noise_robustness/success_rate_{noise_level}": metrics['success_rate'],
                    f"noise_robustness/overlap_improvement_{noise_level}": metrics['avg_overlap_improvement']
                })
            
            # Create and log noise robustness plot
            self.visualizer.log_noise_robustness(results, pattern_type)
        
        return results
    
    def run_convergence_experiment(self, pattern_type: str = "simple_shapes") -> Dict[str, List[float]]:
        """Wrapper for convergence dynamics experiment with W&B logging."""
        results = self.experiment_convergence_dynamics()
        
        # Log results to W&B
        if self.visualizer:
            # Calculate summary statistics
            avg_convergence_steps = np.mean(results['convergence_steps'])
            avg_energy_decrease = np.mean(results['energy_decreases'])
            avg_overlap_improvement = np.mean([
                f - i for i, f in zip(results['initial_overlaps'], results['final_overlaps'])
            ])
            
            self.visualizer.log_metrics({
                "convergence/avg_steps": avg_convergence_steps,
                "convergence/avg_energy_decrease": avg_energy_decrease,
                "convergence/avg_overlap_improvement": avg_overlap_improvement,
                "convergence/total_trials": len(results['convergence_steps'])
            })
            
            # Create and log convergence analysis
            self.visualizer.log_convergence_analysis(results['convergence_steps'], "convergence_dynamics")
        
        return results
    
    def create_comprehensive_visualizations(self) -> None:
        """Create comprehensive visualizations for all experiments."""
        if not self.experiment_results:
            logger.warning("No experiment results available for visualization")
            return
        
        # Create summary comparison plots
        if self.visualizer:
            create_comprehensive_comparison(self.experiment_results)
        
        logger.info("Comprehensive visualizations created")
    
    def save_trained_model(self) -> None:
        """Save the trained model weights and configuration."""
        model_path = Path(MODELS_DIR) / 'hopfield_network.npz'
        
        # Save model data
        model_data = {
            'weights': self.network.weights,
            'network_size': self.network_size,
            'stored_patterns': np.array(self.network.stored_patterns) if self.network.stored_patterns else np.array([]),
            'random_seed': RANDOM_SEED if USE_FIXED_SEED else None
        }
        
        np.savez_compressed(model_path, **model_data)
        logger.info(f"Model saved to {model_path}")
        
        # Log model to W&B if available
        if self.visualizer:
            self.visualizer.log_file_artifact(
                str(model_path),
                "trained_model",
                "Trained Hopfield Network weights and configuration"
            )
    
    def generate_experiment_report(self) -> None:
        """Generate comprehensive experimental report."""
        self._generate_experimental_report()
        
        # Log report to W&B if available
        if self.visualizer:
            report_path = Path(OUTPUT_DIR) / 'hopfield_experimental_report.txt'
            if report_path.exists():
                self.visualizer.log_file_artifact(
                    str(report_path), 
                    "experiment_report", 
                    "Comprehensive experimental report"
                )
    
    def train_basic_patterns_with_logging(self, pattern_type: str = "simple_shapes") -> Dict[str, float]:
        """
        Train the network on basic patterns with W&B logging.
        
        Args:
            pattern_type: Type of patterns to use ('simple_shapes', 'letters', 'random')
            
        Returns:
            Dictionary with training results
            
        Educational Focus:
            Demonstrates the basic Hebbian learning process and immediate
            storage without iterative training (contrast with backpropagation).
        """
        logger.info(f"Starting basic pattern training with {pattern_type} (W&B logging)")
        
        # Load patterns
        patterns = self.data_loader.load_pattern_set(pattern_type)
        pattern_list = list(patterns.values())
        
        # Limit to maximum patterns for this experiment
        if len(pattern_list) > MAX_PATTERNS:
            pattern_list = pattern_list[:MAX_PATTERNS]
            logger.info(f"Limited to {MAX_PATTERNS} patterns for capacity reasons")
        
        # Store patterns (one-shot learning!)
        start_time = time.time()
        self.network.store_patterns(pattern_list)
        training_time = time.time() - start_time
        
        logger.info(f"Pattern storage completed in {training_time:.4f} seconds")
        
        # Evaluate storage quality
        results = self._evaluate_pattern_storage(patterns, pattern_list)
        results['training_time'] = training_time
        results['pattern_type'] = pattern_type
        results['num_patterns'] = len(pattern_list)
        
        # Save results
        self.experiment_results['basic_training'] = results
        
        # Log results to W&B
        if self.visualizer:
            self.visualizer.log_metrics({
                "basic_training/perfect_recall_rate": results['perfect_recall_rate'],
                "basic_training/avg_energy_decrease": results['avg_energy_decrease'],
                "basic_training/avg_convergence_steps": results['avg_convergence_steps'],
                "basic_training/training_time": training_time
            })
        
        # Log pattern visualization
        pattern_plot_path = Path(PLOTS_DIR) / f"stored_patterns_{pattern_type}.png"
        self.data_loader.generator.visualize_pattern_set(
            patterns, 
            f"Stored {pattern_type} Patterns",
            save_path=pattern_plot_path
        )
        if self.visualizer:
            self.visualizer.log_image(str(pattern_plot_path), f"patterns/{pattern_type}_stored")
        
        logger.info(f"Basic training results: {results}")
        return results

    @staticmethod
    def corrupt_pattern(pattern: np.ndarray, num_flips: int = 5) -> np.ndarray:
        """
        Flips a specified number of bits in a pattern for testing recall.

        Args:
            pattern: The original pattern to corrupt
            num_flips: The number of bits to flip

        Returns:
            The corrupted pattern with flipped bits
            
        Educational Focus:
            Simulates real-world noise in stored memories to test
            the network's error correction capabilities.
        """
        corrupted = np.copy(pattern)
        # Choose random indices to flip without replacement
        flip_indices = np.random.choice(len(pattern), size=num_flips, replace=False)
        # Flip the bits at the chosen indices (-1 becomes 1, and 1 becomes -1)
        corrupted[flip_indices] *= -1
        return corrupted

    def run_interactive_demo(self) -> Dict[str, Any]:
        """
        Run an interactive console demonstration of the Hopfield Network.
        
        This provides immediate visual feedback for educational purposes,
        showing the complete pattern storage and recall process step-by-step.
        
        Returns:
            Dictionary with demo results
            
        Educational Focus:
            Interactive learning experience showing:
            - Pattern storage (Hebbian learning)
            - Noise corruption simulation
            - Associative memory recall
            - Capacity limitations with increasing patterns
        """
        from .visualize import display_pattern
        
        logger.info("Starting Interactive Hopfield Network Demonstration")
        print("\n" + "="*60)
        print("HOPFIELD NETWORK INTERACTIVE DEMONSTRATION")
        print("="*60)
        
        # Load educational letter patterns directly from the generator
        patterns_dict = self.data_loader.generator.get_educational_letters()
        all_patterns_list = list(patterns_dict.values())
        all_patterns_names = list(patterns_dict.keys())
        
        demo_results = {
            'single_recall_success': False,
            'capacity_test_results': [],
            'total_patterns_tested': len(all_patterns_list)
        }
        
        # --- Single Pattern Recall Demonstration ---
        print("\n--- Single Pattern Recall Demonstration ---")
        network = HopfieldNetwork(size=len(all_patterns_list[0]))
        
        # Train on the first 3 patterns
        training_patterns = all_patterns_list[:3]
        network.store_patterns(training_patterns)
        print(f"Stored 3 patterns: {all_patterns_names[:3]}")
        
        # Demonstrate recall with pattern 'C'
        original_pattern_name = 'C'
        original_pattern = patterns_dict[original_pattern_name]
        
        print(f"\nOriginal Pattern '{original_pattern_name}':")
        display_pattern(original_pattern, title=f"Original Pattern '{original_pattern_name}'")
        
        # Corrupt the pattern
        corrupted = self.corrupt_pattern(original_pattern, num_flips=5)
        print("\nCorrupted Pattern (5 bits flipped):")
        display_pattern(corrupted, title="Corrupted Pattern")
        
        # Recall the pattern
        recalled_pattern, _ = network.retrieve_pattern(corrupted)
        print("\nRecalled Pattern:")
        display_pattern(recalled_pattern, title="Recalled Pattern")
        
        # Check if recall was successful
        demo_results['single_recall_success'] = np.array_equal(recalled_pattern, original_pattern)
        success_msg = "✓ SUCCESS!" if demo_results['single_recall_success'] else "✗ FAILED"
        print(f"\nRecall Result: {success_msg}")
        
        print("\n" + "="*40 + "\n")
        
        # --- Network Capacity Test Demonstration ---
        print("--- Network Capacity Test ---")
        print("Testing how many patterns the network can reliably store...")
        
        original_pattern_to_test = patterns_dict['C']
        corrupted_to_test = self.corrupt_pattern(original_pattern_to_test, num_flips=5)
        
        # Test capacity with increasing number of patterns
        for num_stored in range(1, len(all_patterns_list) + 1):
            print(f"\n--- Testing with {num_stored} stored pattern(s) ---")
            patterns_to_store = all_patterns_list[:num_stored]
            pattern_names = all_patterns_names[:num_stored]
            
            # Create fresh network for each test
            test_network = HopfieldNetwork(size=len(all_patterns_list[0]))
            test_network.store_patterns(patterns_to_store)
            print(f"Storing: {pattern_names}")
            
            # Attempt to recall the corrupted 'C'
            recalled, _ = test_network.retrieve_pattern(corrupted_to_test)
            
            # Check success
            recall_successful = np.array_equal(recalled, original_pattern_to_test)
            demo_results['capacity_test_results'].append({
                'num_patterns': num_stored,
                'pattern_names': pattern_names.copy(),
                'recall_successful': recall_successful
            })
            
            if recall_successful:
                print("✓ Recall successful!")
                display_pattern(recalled, title="Recalled 'C' Correctly")
            else:
                print("✗ Recall FAILED.")
                display_pattern(recalled, title="Incorrectly Recalled Pattern")
                print(f"Network capacity exceeded at {num_stored} patterns.")
                break
        
        # Summary
        successful_recalls = sum(1 for result in demo_results['capacity_test_results'] 
                               if result['recall_successful'])
        print(f"\n" + "="*60)
        print("DEMONSTRATION SUMMARY")
        print("="*60)
        print(f"Single pattern recall: {'SUCCESS' if demo_results['single_recall_success'] else 'FAILED'}")
        print(f"Maximum capacity demonstrated: {successful_recalls} patterns")
        print(f"Theoretical capacity (0.138 * N): ~{int(0.138 * len(all_patterns_list[0]))} patterns")
        print("="*60)
        
        # Log results to W&B if available
        if self.visualizer:
            self.visualizer.log_metrics({
                "demo/single_recall_success": 1.0 if demo_results['single_recall_success'] else 0.0,
                "demo/max_capacity_demonstrated": successful_recalls,
                "demo/theoretical_capacity": int(0.138 * len(all_patterns_list[0])),
                "demo/total_patterns_available": len(all_patterns_list)
            })
        
        return demo_results

    def run_mnist_demonstration(self, wandb_visualizer=None) -> Dict[str, Any]:
        """
        Demonstrate Hopfield network storage and retrieval with MNIST digits.
        
        This demonstrates the network's capabilities and limitations when scaling
        to realistic datasets like MNIST handwritten digits.
        
        Args:
            wandb_visualizer: Optional W&B visualizer for logging results
            
        Returns:
            Dictionary with MNIST demonstration results
            
        Educational Focus:
            - Real-world data challenges for associative memory
            - Storage capacity limitations with high-dimensional patterns
            - Pattern correlation effects on memory retrieval
            - Comparison with modern deep learning approaches
        """
        logger.info("Starting MNIST demonstration...")
        
        # Import MNIST functions here to avoid top-level imports
        from .mnist_demo import (
            create_synthetic_mnist, preprocess_mnist_for_hopfield,
            select_representative_digits, test_pattern_retrieval,
            add_noise_to_patterns, create_mnist_visualization,
            print_mnist_summary
        )
        
        # Create synthetic MNIST-like data
        train_images, train_labels, test_images, test_labels = create_synthetic_mnist()
        
        # Preprocess for Hopfield network
        train_binary = preprocess_mnist_for_hopfield(train_images)
        test_binary = preprocess_mnist_for_hopfield(test_images)
        
        # Select representative digits (1 example per class = 10 total patterns)
        stored_patterns, stored_labels = select_representative_digits(
            train_binary, train_labels, digits_per_class=1
        )
        
        logger.info(f"Selected {len(stored_patterns)} patterns for storage")
        
        # Create and configure Hopfield network for MNIST size
        network = HopfieldNetwork(size=784)  # 28x28 = 784 neurons
        
        # Store the patterns
        logger.info("Storing patterns in Hopfield network...")
        network.store_patterns(stored_patterns)
        
        # Test retrieval with the stored patterns themselves
        logger.info("Testing perfect retrieval...")
        perfect_results = test_pattern_retrieval(network, stored_patterns, stored_labels, "Perfect")
        
        # Test with noisy versions
        logger.info("Testing noisy retrieval...")
        noisy_patterns = add_noise_to_patterns(stored_patterns, noise_level=0.1)
        noisy_results = test_pattern_retrieval(network, noisy_patterns, stored_labels, "Noisy (10%)")
        
        # Test with test set examples
        logger.info("Testing with unseen examples...")
        test_patterns, test_digit_labels = select_representative_digits(
            test_binary, test_labels, digits_per_class=5
        )
        test_results = test_pattern_retrieval(network, test_patterns, test_digit_labels, "Unseen")
        
        # Create visualizations
        create_mnist_visualization(stored_patterns, stored_labels, perfect_results, 
                                 noisy_results, test_results, wandb_visualizer)
        
        # Log results to W&B if available
        if self.visualizer:
            self.visualizer.log_metrics({
                "mnist/perfect_success_rate": perfect_results['successful_retrievals'] / perfect_results['total_tests'],
                "mnist/noisy_success_rate": noisy_results['successful_retrievals'] / noisy_results['total_tests'], 
                "mnist/test_success_rate": test_results['successful_retrievals'] / test_results['total_tests'],
                "mnist/network_size": 784,
                "mnist/stored_patterns": len(stored_patterns),
                "mnist/theoretical_capacity": int(0.15 * 784)
            })
        
        # Print comprehensive summary
        print_mnist_summary(perfect_results, noisy_results, test_results, len(stored_patterns))
        
        return {
            'perfect_results': perfect_results,
            'noisy_results': noisy_results,
            'test_results': test_results
        }

    def create_shifted_patterns(self, pattern: np.ndarray, shifts: List[Tuple[int, int]]) -> List[np.ndarray]:
        """
        Create shifted versions of a pattern to test spatial invariance.
        
        Args:
            pattern: Original binary pattern (flattened)
            shifts: List of (row_shift, col_shift) tuples
            
        Returns:
            List of shifted patterns
            
        Educational Focus:
            Demonstrates why Hopfield networks fail with spatial translations,
            providing motivation for convolutional architectures.
        """
        # Reshape to 2D for shifting
        size = int(np.sqrt(len(pattern)))
        pattern_2d = pattern.reshape(size, size)
        
        shifted_patterns = []
        
        for row_shift, col_shift in shifts:
            # Create shifted pattern with zero padding
            shifted = np.zeros_like(pattern_2d)
            
            # Calculate valid regions
            src_row_start = max(0, -row_shift)
            src_row_end = min(size, size - row_shift)
            src_col_start = max(0, -col_shift)
            src_col_end = min(size, size - col_shift)
            
            dst_row_start = max(0, row_shift)
            dst_row_end = dst_row_start + (src_row_end - src_row_start)
            dst_col_start = max(0, col_shift)
            dst_col_end = dst_col_start + (src_col_end - src_col_start)
            
            # Copy valid region
            if src_row_end > src_row_start and src_col_end > src_col_start:
                shifted[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = \
                    pattern_2d[src_row_start:src_row_end, src_col_start:src_col_end]
            
            shifted_patterns.append(shifted.flatten())
        
        return shifted_patterns

    def run_spatial_invariance_demo(self) -> Dict[str, Any]:
        """
        Demonstrate that Hopfield networks cannot handle spatial translations.
        
        This is an important educational demonstration showing why position-specific
        associative memory is insufficient for real-world pattern recognition.
        
        Returns:
            Dictionary with spatial invariance test results
            
        Educational Focus:
            Shows fundamental limitations that motivated convolutional architectures:
            - Position-specific pattern storage
            - Failure with spatial translations
            - Motivation for weight sharing in CNNs
        """
        logger.info("Running spatial invariance limitation demonstration...")
        
        # Create network with appropriate size for simple patterns
        network = HopfieldNetwork(size=100)  # 10x10 patterns
        
        # Create a simple digit pattern (e.g., a cross shape)
        from .data_loader import create_simple_digit
        original_pattern = create_simple_digit('cross', size=10)
        
        # Store the original pattern
        network.store_patterns([original_pattern])
        logger.info(f"Stored original pattern at center position")
        
        # Test shifts
        shifts = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (-1, 0), (0, -1)]
        shift_labels = ['Original', 'Down 1', 'Right 1', 'Down+Right 1', 
                       'Down 2', 'Right 2', 'Up 1', 'Left 1']
        
        # Create shifted versions
        shifted_patterns = self.create_shifted_patterns(original_pattern, shifts)
        
        # Test retrieval for each shifted pattern
        results = []
        overlaps = []
        
        for i, (shifted_pattern, label) in enumerate(zip(shifted_patterns, shift_labels)):
            # Retrieve using shifted pattern as input
            retrieved, retrieval_info = network.retrieve_pattern(shifted_pattern)
            
            # Calculate overlap with original stored pattern
            final_overlap = np.mean(retrieved == original_pattern)
            
            result = {
                'shift': shifts[i],
                'label': label,
                'input_pattern': shifted_pattern,
                'retrieved_pattern': retrieved,
                'overlap_with_original': final_overlap,
                'retrieval_successful': final_overlap > 0.8,
                'convergence_steps': retrieval_info.get('steps', 0)
            }
            
            results.append(result)
            overlaps.append(final_overlap)
            
            logger.info(f"{label}: overlap = {final_overlap:.3f}, "
                       f"successful = {final_overlap > 0.8}")
        
        # Create visualization
        from .visualize import plot_spatial_invariance_results, print_spatial_educational_summary
        plot_spatial_invariance_results(results, shift_labels, overlaps)
        
        # Print educational summary
        print_spatial_educational_summary(results)
        
        # Log results to W&B if available
        if self.visualizer:
            successful_retrievals = sum(1 for r in results if r['retrieval_successful'])
            self.visualizer.log_metrics({
                "spatial_invariance/success_rate": successful_retrievals / len(results),
                "spatial_invariance/original_success": results[0]['retrieval_successful'],
                "spatial_invariance/shifted_failures": len(results) - successful_retrievals - 1,
                "spatial_invariance/avg_overlap": np.mean(overlaps),
                "spatial_invariance/worst_overlap": np.min(overlaps[1:])  # Exclude original
            })
        
        return {
            'test_results': results,
            'success_rate': successful_retrievals / len(results),
            'total_tests': len(results),
            'successful_retrievals': successful_retrievals,
            'average_overlap': np.mean(overlaps)
        }

def parse_arguments():
    """Parse command line arguments for Hopfield Network training."""
    parser = argparse.ArgumentParser(description='Train Hopfield Network - AI From Scratch to Scale')
    
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases experiment tracking'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['all', 'basic', 'capacity', 'noise', 'convergence', 'mnist', 'demo', 'spatial_invariance'],
        default='all',
        help='Which experiment to run (default: all)'
    )
    
    parser.add_argument(
        '--network-size',
        type=int,
        default=NETWORK_SIZE,
        help=f'Size of the Hopfield network (default: {NETWORK_SIZE})'
    )
    
    parser.add_argument(
        '--patterns',
        type=int,
        default=MAX_PATTERNS,
        help=f'Maximum number of patterns to store (default: {MAX_PATTERNS})'
    )
    
    parser.add_argument(
        '--pattern-type',
        type=str,
        choices=['simple_shapes', 'letters', 'random'],
        default='simple_shapes',
        help='Type of patterns to use (default: simple_shapes)'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main training function with W&B integration."""
    args = parse_arguments()
    
    logger.info("="*70)
    logger.info("HOPFIELD NETWORK TRAINING - AI FROM SCRATCH TO SCALE")
    logger.info("="*70)
    
    # Initialize W&B
    config = {
        "network_size": args.network_size,
        "max_patterns": args.patterns,
        "pattern_type": args.pattern_type,
        "experiment": args.experiment,
        "random_seed": RANDOM_SEED if USE_FIXED_SEED else None
    }
    
    wandb_run, visualizer = initialize_wandb(
        project_name=WANDB_PROJECT_NAME,
        config=config,
        enabled=not args.no_wandb
    )
    
    try:
        # Initialize trainer with W&B visualizer
        trainer = HopfieldTrainer(
            network_size=args.network_size,
            wandb_visualizer=visualizer
        )
        
        # Run experiments based on arguments
        if args.experiment == 'all':
            logger.info("Running comprehensive experiment suite...")
            
            # 1. Basic pattern training
            logger.info("1. Basic Pattern Training")
            basic_results = trainer.train_basic_patterns(args.pattern_type)
            if visualizer:
                visualizer.log_experiment_results("basic_training", basic_results)
            
            # 2. Storage capacity analysis
            logger.info("2. Storage Capacity Analysis")
            capacity_results = trainer.run_capacity_experiment()
            if visualizer:
                visualizer.log_experiment_results("capacity_analysis", capacity_results)
            
            # 3. Noise robustness testing
            logger.info("3. Noise Robustness Testing")
            noise_results = trainer.run_noise_robustness_experiment(args.pattern_type)
            if visualizer:
                visualizer.log_experiment_results("noise_robustness", noise_results)
            
            # 4. Convergence dynamics
            logger.info("4. Convergence Dynamics Analysis")
            convergence_results = trainer.run_convergence_experiment(args.pattern_type)
            if visualizer:
                visualizer.log_experiment_results("convergence_analysis", convergence_results)
            
            # 5. Create comprehensive visualizations
            logger.info("5. Creating Visualizations")
            trainer.create_comprehensive_visualizations()
            
        elif args.experiment == 'basic':
            results = trainer.train_basic_patterns(args.pattern_type)
            if visualizer:
                visualizer.log_experiment_results("basic_training", results)
            
        elif args.experiment == 'capacity':
            results = trainer.run_capacity_experiment()
            if visualizer:
                visualizer.log_experiment_results("capacity_analysis", results)
            
        elif args.experiment == 'noise':
            results = trainer.run_noise_robustness_experiment(args.pattern_type)
            if visualizer:
                visualizer.log_experiment_results("noise_robustness", results)
            
        elif args.experiment == 'convergence':
            results = trainer.run_convergence_experiment(args.pattern_type)
            if visualizer:
                visualizer.log_experiment_results("convergence_analysis", results)
            
        elif args.experiment == 'mnist':
            logger.info("Running MNIST demonstration...")
            mnist_results = trainer.run_mnist_demonstration(visualizer)
            if visualizer:
                visualizer.log_experiment_results("mnist_demonstration", mnist_results)
        
        elif args.experiment == 'demo':
            logger.info("Running interactive demonstration...")
            demo_results = trainer.run_interactive_demo()
            if visualizer:
                visualizer.log_experiment_results("interactive_demo", demo_results)
        
        elif args.experiment == 'spatial_invariance':
            logger.info("Running spatial invariance demonstration...")
            spatial_results = trainer.run_spatial_invariance_demo()
            if visualizer:
                visualizer.log_experiment_results("spatial_invariance_demo", spatial_results)
        
        # Create experiment summary for W&B
        if visualizer and trainer.experiment_results:
            visualizer.create_experiment_summary(trainer.experiment_results)
        
        # Save model artifact
        if visualizer:
            model_state = {
                'weights': trainer.network.weights,
                'stored_patterns': np.array(trainer.network.stored_patterns) if trainer.network.stored_patterns else np.array([]),
                'network_size': trainer.network_size,
                'config': config
            }
            visualizer.save_model_artifact(model_state, f"hopfield_model_{args.experiment}")
        
        # Generate final report
        trainer.generate_experiment_report()
        trainer.save_trained_model()
        
        logger.info("="*70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"Results saved to: {OUTPUT_DIR}")
        if not args.no_wandb:
            logger.info(f"W&B dashboard: https://wandb.ai/project/{WANDB_PROJECT_NAME}")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Finish W&B run
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
