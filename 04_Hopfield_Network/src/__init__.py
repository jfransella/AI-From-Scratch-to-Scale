# -*- coding: utf-8 -*-
"""Hopfield Network implementation package.

This package provides a from-scratch implementation of the Hopfield Network,
a form of recurrent artificial neural network that serves as content-addressable
memory. It includes:

- Core Hopfield Network model with pattern storage and retrieval
- Data loaders for pattern datasets and memory experiments
- Visualization functions for network states and energy landscapes
- Weights & Biases integration for experiment tracking
- Configuration management for network parameters

The implementation demonstrates associative memory principles and energy-based
learning without framework dependencies.

Example:
    >>> from src import HopfieldNetwork, HopfieldWandbVisualizer
    >>> network = HopfieldNetwork(n_neurons=100)
    >>> network.fit(patterns)
    >>> retrieved = network.predict(noisy_pattern)
"""

from .model import HopfieldNetwork
# from .wandb_integration import HopfieldWandbVisualizer  # Temporarily disabled - WandB integration pending

# Core visualization functions - public API
from .visualize import (
    display_pattern,
    visualize_pattern_set,
    visualize_energy_landscape,
    visualize_convergence,
    plot_capacity_results,
    plot_noise_robustness,
    plot_convergence_statistics,
    HopfieldVisualizer
)

__all__ = [
    # Core model
    'HopfieldNetwork',
    
    # Experiment tracking
    # 'HopfieldWandbVisualizer',  # Temporarily disabled - WandB integration pending
    
    # Visualization functions
    'display_pattern',
    'visualize_pattern', 
    'visualize_pattern_set',
    'visualize_energy_landscape',
    'visualize_convergence',
    'plot_capacity_results',
    'plot_noise_robustness',
    'plot_convergence_statistics',
]

__version__ = "1.0.0"
__author__ = "AI-From-Scratch-to-Scale Project"
