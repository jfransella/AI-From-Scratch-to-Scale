"""
Hopfield Network: Energy-Based Associative Memory Model
======================================================

This module implements the classic Hopfield Network (1982), demonstrating
energy-based learning as an alternative paradigm to gradient-based methods.

Key Educational Concepts:
- Energy functions and Lyapunov stability
- Symmetric weight matrices and convergence guarantees  
- Associative memory vs. supervised classification
- Statistical mechanics analogies (Ising model, spin glasses)
- Content-addressable memory systems

Mathematical Foundation:
- Energy: E = -0.5 * Σ(i,j) w_ij * s_i * s_j
- Update rule: s_i = sign(Σ(j) w_ij * s_j)
- Hebbian storage: w_ij = (1/N) * Σ(μ) ξ_i^μ * ξ_j^μ

Historical Context:
John Hopfield (1982) showed how neural networks could function as
content-addressable memory, bridging neuroscience and physics.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
from pathlib import Path

from .config import (
    NETWORK_SIZE, PATTERN_ON, PATTERN_OFF, UPDATE_RULE, MAX_ITERATIONS,
    CONVERGENCE_THRESHOLD, TEMPERATURE, USE_TEMPERATURE, ZERO_DIAGONAL,
    SYMMETRIC_WEIGHTS, LEARNING_RULE, OVERLAP_THRESHOLD, MAX_RETRIEVAL_STEPS,
    PLOTS_DIR, MODELS_DIR, RANDOM_SEED, USE_FIXED_SEED
)

# Set up logging
logger = logging.getLogger(__name__)


class HopfieldNetwork:
    """
    Hopfield Network implementation focusing on educational clarity.
    
    This implementation emphasizes understanding the energy-based paradigm
    and its fundamental differences from gradient-based learning methods.
    """
    
    def __init__(self, size: int = NETWORK_SIZE):
        """
        Initialize Hopfield Network.
        
        Args:
            size: Number of neurons in the network
        """
        self.size = size
        self.weights = np.zeros((size, size), dtype=np.float64)
        self.state = np.zeros(size, dtype=int)
        self.stored_patterns = []
        self.energy_history = []
        self.convergence_history = []
        
        if USE_FIXED_SEED:
            np.random.seed(RANDOM_SEED)
        
        logger.info(f"Hopfield Network initialized with {size} neurons")
    
    def _calculate_energy(self, state: np.ndarray) -> float:
        """
        Calculate the energy of a given state.
        
        Args:
            state: Network state vector
            
        Returns:
            Energy value
            
        Mathematical Definition:
            E = -0.5 * Σ(i,j) w_ij * s_i * s_j
            
        Note:
            The factor of 0.5 prevents double-counting since w_ij = w_ji
        """
        energy = -0.5 * np.sum(self.weights * np.outer(state, state))
        return energy
    
    def _calculate_local_field(self, neuron_idx: int, state: np.ndarray) -> float:
        """
        Calculate local field (net input) for a specific neuron.
        
        Args:
            neuron_idx: Index of neuron
            state: Current network state
            
        Returns:
            Local field value
            
        Mathematical Definition:
            h_i = Σ(j≠i) w_ij * s_j
        """
        local_field = np.sum(self.weights[neuron_idx, :] * state)
        return local_field
    
    def _update_neuron(self, neuron_idx: int, state: np.ndarray) -> int:
        """
        Update a single neuron using the Hopfield update rule.
        
        Args:
            neuron_idx: Index of neuron to update
            state: Current network state
            
        Returns:
            New state for the neuron
            
        Mathematical Rule:
            s_i = sign(h_i) = sign(Σ(j) w_ij * s_j)
        """
        local_field = self._calculate_local_field(neuron_idx, state)
        
        if USE_TEMPERATURE and TEMPERATURE > 0:
            # Stochastic update with temperature
            probability = 1.0 / (1.0 + np.exp(-2 * local_field / TEMPERATURE))
            new_state = PATTERN_ON if np.random.random() < probability else PATTERN_OFF
        else:
            # Deterministic update
            new_state = PATTERN_ON if local_field > 0 else PATTERN_OFF
        
        return new_state
    
    def store_pattern(self, pattern: np.ndarray) -> None:
        """
        Store a single pattern using Hebbian learning.
        
        Args:
            pattern: Binary pattern to store
            
        Mathematical Rule:
            Δw_ij = η * ξ_i * ξ_j (for i ≠ j)
            
        Note:
            For multiple patterns: w_ij = (1/N) * Σ(μ) ξ_i^μ * ξ_j^μ
        """
        if len(pattern) != self.size:
            raise ValueError(f"Pattern size {len(pattern)} doesn't match network size {self.size}")
        
        # Hebbian learning rule: outer product of pattern with itself
        weight_update = np.outer(pattern, pattern)
        
        if ZERO_DIAGONAL:
            np.fill_diagonal(weight_update, 0)  # No self-connections
        
        self.weights += weight_update
        self.stored_patterns.append(pattern.copy())
        
        logger.debug(f"Stored pattern {len(self.stored_patterns)}")
    
    def store_patterns(self, patterns: List[np.ndarray]) -> None:
        """
        Store multiple patterns in the network.
        
        Args:
            patterns: List of binary patterns to store
            
        Mathematical Implementation:
            w_ij = (1/N) * Σ(μ=1 to P) ξ_i^μ * ξ_j^μ
            where P is the number of patterns
        """
        if LEARNING_RULE == "hebbian":
            self._store_patterns_hebbian(patterns)
        elif LEARNING_RULE == "pseudoinverse":
            self._store_patterns_pseudoinverse(patterns)
        else:
            raise ValueError(f"Unknown learning rule: {LEARNING_RULE}")
        
        if SYMMETRIC_WEIGHTS:
            self.weights = 0.5 * (self.weights + self.weights.T)
        
        if ZERO_DIAGONAL:
            np.fill_diagonal(self.weights, 0)
        
        logger.info(f"Stored {len(patterns)} patterns using {LEARNING_RULE} rule")
    
    def _store_patterns_hebbian(self, patterns: List[np.ndarray]) -> None:
        """
        Store patterns using standard Hebbian rule.
        
        Args:
            patterns: List of patterns to store
        """
        self.weights.fill(0)  # Reset weights
        
        for pattern in patterns:
            if len(pattern) != self.size:
                raise ValueError(f"Pattern size {len(pattern)} doesn't match network size {self.size}")
            
            # Accumulate outer products
            self.weights += np.outer(pattern, pattern)
            self.stored_patterns.append(pattern.copy())
        
        # Normalize by number of patterns
        self.weights /= len(patterns)
    
    def _store_patterns_pseudoinverse(self, patterns: List[np.ndarray]) -> None:
        """
        Store patterns using pseudoinverse rule (better capacity).
        
        Args:
            patterns: List of patterns to store
            
        Mathematical Foundation:
            W = P * (P^T * P)^(-1) * P^T
            where P is the pattern matrix
        """
        pattern_matrix = np.array(patterns)  # Shape: (num_patterns, size)
        
        try:
            # Pseudoinverse approach
            gram_matrix = pattern_matrix @ pattern_matrix.T
            weights = pattern_matrix.T @ np.linalg.pinv(gram_matrix) @ pattern_matrix
            self.weights = weights
            self.stored_patterns = [p.copy() for p in patterns]
        except np.linalg.LinAlgError:
            logger.warning("Pseudoinverse failed, falling back to Hebbian rule")
            self._store_patterns_hebbian(patterns)
    
    def update_asynchronous(self, state: np.ndarray, max_iter: int = MAX_ITERATIONS) -> Tuple[np.ndarray, List[float]]:
        """
        Update network state asynchronously (one neuron at a time).
        
        Args:
            state: Initial state
            max_iter: Maximum number of iterations
            
        Returns:
            Tuple of (final_state, energy_history)
            
        Educational Note:
            Asynchronous updates guarantee convergence to a fixed point
            because the energy function is a Lyapunov function.
        """
        current_state = state.copy()
        energy_history = [self._calculate_energy(current_state)]
        
        for iteration in range(max_iter):
            # Randomly select neuron to update
            neuron_idx = np.random.randint(self.size)
            
            # Update selected neuron
            new_value = self._update_neuron(neuron_idx, current_state)
            old_value = current_state[neuron_idx]
            current_state[neuron_idx] = new_value
            
            # Calculate new energy
            current_energy = self._calculate_energy(current_state)
            energy_history.append(current_energy)
            
            # Check for convergence
            if len(energy_history) > 1:
                energy_change = abs(energy_history[-1] - energy_history[-2])
                if energy_change < CONVERGENCE_THRESHOLD:
                    logger.debug(f"Converged after {iteration + 1} iterations")
                    break
            
            # Verify energy never increases (Lyapunov property)
            if len(energy_history) > 1 and energy_history[-1] > energy_history[-2] + 1e-10:
                logger.warning(f"Energy increased at iteration {iteration}! This violates Lyapunov property.")
        
        return current_state, energy_history
    
    def update_synchronous(self, state: np.ndarray, max_iter: int = MAX_ITERATIONS) -> Tuple[np.ndarray, List[float]]:
        """
        Update network state synchronously (all neurons simultaneously).
        
        Args:
            state: Initial state
            max_iter: Maximum number of iterations
            
        Returns:
            Tuple of (final_state, energy_history)
            
        Educational Note:
            Synchronous updates may lead to cycles and don't guarantee
            convergence, but they're computationally more efficient.
        """
        current_state = state.copy()
        energy_history = [self._calculate_energy(current_state)]
        
        for iteration in range(max_iter):
            # Calculate local fields for all neurons
            local_fields = self.weights @ current_state
            
            # Update all neurons simultaneously
            if USE_TEMPERATURE and TEMPERATURE > 0:
                # Stochastic updates
                probabilities = 1.0 / (1.0 + np.exp(-2 * local_fields / TEMPERATURE))
                new_state = np.where(np.random.random(self.size) < probabilities, PATTERN_ON, PATTERN_OFF)
            else:
                # Deterministic updates
                new_state = np.where(local_fields > 0, PATTERN_ON, PATTERN_OFF)
            
            # Check for convergence (no change in state)
            if np.array_equal(current_state, new_state):
                logger.debug(f"Converged after {iteration + 1} iterations")
                break
            
            current_state = new_state
            current_energy = self._calculate_energy(current_state)
            energy_history.append(current_energy)
        
        return current_state, energy_history
    
    def retrieve_pattern(self, initial_state: np.ndarray, return_overlap: bool = False) -> Union[Tuple[np.ndarray, Dict[str, Union[int, float, List[float]]]], Tuple[np.ndarray, Dict[str, Union[int, float, List[float]]], float]]:
        """
        Retrieve a stored pattern from an initial (possibly noisy) state.
        
        Args:
            initial_state: Starting state for retrieval
            return_overlap: If True, also return overlap with best stored pattern
            
        Returns:
            Tuple of (final_state, retrieval_info) or (final_state, retrieval_info, overlap)
            
        Educational Purpose:
            Demonstrates the core functionality of associative memory:
            given a partial or corrupted pattern, recover the stored version.
        """
        self.state = initial_state.copy()
        
        if UPDATE_RULE == "asynchronous":
            final_state, energy_history = self.update_asynchronous(self.state, MAX_RETRIEVAL_STEPS)
        elif UPDATE_RULE == "synchronous":
            final_state, energy_history = self.update_synchronous(self.state, MAX_RETRIEVAL_STEPS)
        else:
            raise ValueError(f"Unknown update rule: {UPDATE_RULE}")
        
        # Analyze retrieval success
        retrieval_info = self._analyze_retrieval(initial_state, final_state, energy_history)
        
        # Store history for analysis
        self.energy_history.append(energy_history)
        self.convergence_history.append(retrieval_info)
        
        if return_overlap:
            # Calculate overlap with best stored pattern
            if len(self.stored_patterns) > 0:
                overlaps = [np.mean(final_state * pattern) for pattern in self.stored_patterns]
                best_overlap = max(overlaps)
            else:
                best_overlap = 0.0
            return final_state, retrieval_info, best_overlap
        
        return final_state, retrieval_info
    
    def _analyze_retrieval(self, initial_state: np.ndarray, final_state: np.ndarray, 
                          energy_history: List[float]) -> Dict[str, Union[int, float, List[float]]]:
        """
        Analyze the quality and success of pattern retrieval.
        
        Args:
            initial_state: Starting state
            final_state: Final converged state
            energy_history: Energy values during retrieval
            
        Returns:
            Dictionary with retrieval analysis
        """
        # Find best matching stored pattern
        best_overlap = -1
        best_pattern_idx = -1
        
        for idx, stored_pattern in enumerate(self.stored_patterns):
            overlap = np.mean(final_state * stored_pattern)
            if overlap > best_overlap:
                best_overlap = overlap
                best_pattern_idx = idx
        
        # Calculate metrics
        initial_energy = energy_history[0] if energy_history else 0
        final_energy = energy_history[-1] if energy_history else 0
        energy_decrease = initial_energy - final_energy
        
        retrieval_info = {
            'converged_steps': len(energy_history) - 1,
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'energy_decrease': energy_decrease,
            'best_overlap': best_overlap,
            'best_pattern_idx': best_pattern_idx,
            'successful_retrieval': best_overlap >= OVERLAP_THRESHOLD,
            'energy_history': energy_history
        }
        
        return retrieval_info
    
    def calculate_capacity(self, pattern_sizes: List[int], trials: int = 10) -> Dict[int, float]:
        """
        Empirically determine the storage capacity of the network.
        
        Args:
            pattern_sizes: List of number of patterns to test
            trials: Number of trials per pattern size
            
        Returns:
            Dictionary mapping pattern_count -> success_rate
            
        Educational Goal:
            Demonstrate the theoretical capacity limit (~0.15 * N) and
            how performance degrades beyond this limit.
        """
        capacity_results = {}
        
        for num_patterns in pattern_sizes:
            success_rates = []
            
            for trial in range(trials):
                # Generate random patterns
                patterns = []
                for _ in range(num_patterns):
                    pattern = np.random.choice([PATTERN_OFF, PATTERN_ON], size=self.size)
                    patterns.append(pattern)
                
                # Store patterns
                self.weights.fill(0)  # Reset
                self.stored_patterns = []
                self.store_patterns(patterns)
                
                # Test retrieval with noise
                successful_retrievals = 0
                for pattern in patterns:
                    # Add moderate noise
                    noise_level = 0.2
                    noisy_pattern = pattern.copy()
                    num_flips = int(noise_level * len(pattern))
                    if num_flips > 0:
                        flip_idx = np.random.choice(len(pattern), num_flips, replace=False)
                        noisy_pattern[flip_idx] *= -1
                    
                    # Attempt retrieval
                    retrieved, info = self.retrieve_pattern(noisy_pattern)
                    if info['successful_retrieval']:
                        successful_retrievals += 1
                
                success_rate = successful_retrievals / num_patterns
                success_rates.append(success_rate)
            
            capacity_results[num_patterns] = np.mean(success_rates)
            logger.info(f"Capacity test: {num_patterns} patterns -> {capacity_results[num_patterns]:.2f} success rate")
        
        return capacity_results
    
    def visualize_energy_landscape(self, pattern_subset: Optional[List[np.ndarray]] = None, 
                                  num_samples: int = 1000, save_path: Optional[str] = None) -> None:
        """
        Visualize the energy landscape of the network.
        
        Args:
            pattern_subset: Subset of patterns to highlight
            num_samples: Number of random states to sample for energy
            save_path: Optional path to save the figure
            
        Educational Value:
            Shows how stored patterns create low-energy attractors in
            the state space, illustrating the energy-based paradigm.
        """
        if not self.stored_patterns:
            logger.warning("No patterns stored. Cannot visualize energy landscape.")
            return
        
        # Sample random states and calculate energies
        random_energies = []
        for _ in range(num_samples):
            random_state = np.random.choice([PATTERN_OFF, PATTERN_ON], size=self.size)
            energy = self._calculate_energy(random_state)
            random_energies.append(energy)
        
        # Calculate energies of stored patterns
        stored_energies = []
        for pattern in self.stored_patterns:
            energy = self._calculate_energy(pattern)
            stored_energies.append(energy)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot random state energies as histogram
        plt.hist(random_energies, bins=50, alpha=0.7, density=True, 
                label='Random States', color='lightblue')
        
        # Plot stored pattern energies as vertical lines
        for i, energy in enumerate(stored_energies):
            plt.axvline(energy, color='red', linestyle='--', alpha=0.8,
                       label='Stored Patterns' if i == 0 else "")
        
        plt.xlabel('Energy', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Energy Landscape of Hopfield Network', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text with network info
        info_text = f"Network Size: {self.size}\nStored Patterns: {len(self.stored_patterns)}"
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Energy landscape saved to {save_path}")
        
        plt.show()
    
    def visualize_convergence(self, energy_history: List[float], 
                             title: str = "Energy Convergence", save_path: Optional[str] = None) -> None:
        """
        Visualize the energy convergence during pattern retrieval.
        
        Args:
            energy_history: List of energy values over time
            title: Title for the plot
            save_path: Optional path to save the figure
            
        Educational Purpose:
            Demonstrates the Lyapunov property - energy always decreases
            during asynchronous updates, guaranteeing convergence.
        """
        plt.figure(figsize=(10, 6))
        
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
        
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model (weights and stored patterns).
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'weights': self.weights,
            'stored_patterns': np.array(self.stored_patterns),
            'size': self.size,
            'config': {
                'update_rule': UPDATE_RULE,
                'learning_rule': LEARNING_RULE,
                'zero_diagonal': ZERO_DIAGONAL,
                'symmetric_weights': SYMMETRIC_WEIGHTS
            }
        }
        
        np.savez(filepath, **model_data)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a previously saved model.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = np.load(filepath, allow_pickle=True)
        
        self.weights = model_data['weights']
        self.stored_patterns = list(model_data['stored_patterns'])
        self.size = int(model_data['size'])
        
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Loaded {len(self.stored_patterns)} stored patterns")
    
    def get_network_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get comprehensive statistics about the network state.
        
        Returns:
            Dictionary with network statistics
        """
        stats = {
            'network_size': self.size,
            'num_stored_patterns': len(self.stored_patterns),
            'theoretical_capacity': int(0.15 * self.size),
            'weight_matrix_norm': np.linalg.norm(self.weights),
            'weight_symmetry_error': np.linalg.norm(self.weights - self.weights.T),
            'diagonal_norm': np.linalg.norm(np.diag(self.weights)),
        }
        
        if self.stored_patterns:
            # Pattern overlap statistics
            pattern_matrix = np.array(self.stored_patterns)
            overlaps = []
            for i in range(len(self.stored_patterns)):
                for j in range(i+1, len(self.stored_patterns)):
                    overlap = np.mean(pattern_matrix[i] * pattern_matrix[j])
                    overlaps.append(abs(overlap))
            
            stats['mean_pattern_overlap'] = np.mean(overlaps) if overlaps else 0.0
            stats['max_pattern_overlap'] = np.max(overlaps) if overlaps else 0.0
        
        return stats
    
    def __repr__(self) -> str:
        """String representation of the network."""
        return (f"HopfieldNetwork(size={self.size}, "
                f"stored_patterns={len(self.stored_patterns)}, "
                f"update_rule='{UPDATE_RULE}')")


def demonstrate_hopfield_network():
    """
    Comprehensive demonstration of Hopfield Network capabilities.
    
    This function serves as both a test and an educational walkthrough
    of the key concepts in energy-based associative memory.
    """
    print("="*80)
    print("HOPFIELD NETWORK DEMONSTRATION")
    print("="*80)
    
    # Initialize network
    network = HopfieldNetwork(size=100)  # 10x10 grid
    
    print(f"\n1. Network Initialization")
    print(f"Network: {network}")
    
    # Create simple patterns
    print(f"\n2. Creating Test Patterns")
    patterns = []
    
    # Pattern 1: Cross
    cross = np.full(100, PATTERN_OFF)
    cross[40:60] = PATTERN_ON  # Horizontal line (middle rows)
    cross[4::10] = PATTERN_ON  # Vertical line (middle columns) 
    cross[5::10] = PATTERN_ON
    patterns.append(cross)
    
    # Pattern 2: Checkerboard corners
    checker = np.full(100, PATTERN_OFF)
    for i in range(0, 40, 20):
        for j in range(0, 40, 20):
            idx = i + j//10
            if idx < 100:
                checker[idx:idx+10] = PATTERN_ON
    patterns.append(checker)
    
    print(f"Created {len(patterns)} test patterns")
    
    # Store patterns
    print(f"\n3. Storing Patterns (Hebbian Learning)")
    network.store_patterns(patterns)
    
    stats = network.get_network_statistics()
    print(f"Network statistics after training:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test retrieval with noise
    print(f"\n4. Testing Pattern Retrieval with Noise")
    for i, pattern in enumerate(patterns):
        print(f"\nTesting pattern {i+1}:")
        
        # Add noise
        noise_level = 0.3
        noisy_pattern = pattern.copy()
        num_flips = int(noise_level * len(pattern))
        flip_positions = np.random.choice(len(pattern), num_flips, replace=False)
        noisy_pattern[flip_positions] *= -1
        
        # Calculate initial overlap
        initial_overlap = np.mean(pattern * noisy_pattern)
        print(f"  Initial overlap with noise: {initial_overlap:.3f}")
        
        # Retrieve pattern
        retrieved, info = network.retrieve_pattern(noisy_pattern)
        
        # Calculate final overlap
        final_overlap = np.mean(pattern * retrieved)
        print(f"  Final overlap after retrieval: {final_overlap:.3f}")
        print(f"  Converged in {info['converged_steps']} steps")
        print(f"  Energy decreased by {info['energy_decrease']:.4f}")
        print(f"  Retrieval successful: {info['successful_retrieval']}")
        
        # Visualize convergence
        if info['energy_history']:
            network.visualize_convergence(info['energy_history'], 
                                        f"Pattern {i+1} Retrieval Convergence")
    
    # Visualize energy landscape
    print(f"\n5. Visualizing Energy Landscape")
    network.visualize_energy_landscape()
    
    print(f"\n6. Testing Storage Capacity")
    capacity_results = network.calculate_capacity([1, 3, 5, 8, 12, 15], trials=5)
    
    print(f"Capacity test results:")
    for num_patterns, success_rate in capacity_results.items():
        print(f"  {num_patterns} patterns: {success_rate:.2f} success rate")
    
    print(f"\nTheoretical capacity: ~{int(0.15 * network.size)} patterns")
    print(f"Demonstration complete!")
    print("="*80)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demonstration
    demonstrate_hopfield_network()