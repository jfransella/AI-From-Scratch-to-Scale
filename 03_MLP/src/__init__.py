# -*- coding: utf-8 -*-
"""Multi-Layer Perceptron (MLP) implementation package.

This package provides a from-scratch implementation of the Multi-Layer Perceptron,
extending the basic perceptron to handle non-linearly separable problems. It includes:

- Core MLP model with multiple hidden layers and backpropagation
- Various activation functions (sigmoid, tanh, ReLU)
- Data loaders for classification and regression datasets
- Visualization functions for decision boundaries and training dynamics
- Configuration management for network architecture and hyperparameters

The implementation demonstrates the power of deep learning fundamentals
using only NumPy for educational clarity.

Note: This package still uses the legacy Visualizer class pattern and needs 
refactoring to match the standardized W&B integration pattern.

Example:
    >>> from src import MLP
    >>> model = MLP(layer_sizes=[784, 128, 64, 10], activation='relu')
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
"""

from .model import MLP

# Configuration  
from .config import EXPERIMENTS, WANDB_PROJECT_NAME

# Legacy visualizer (needs refactoring)
from .visualize import Visualizer

__all__ = [
    # Core model
    'MLP',
    
    # Configuration
    'EXPERIMENTS', 
    'WANDB_PROJECT_NAME',
    
    # Legacy visualization (to be refactored)
    'Visualizer',
]

__version__ = "1.0.0"
__author__ = "AI-From-Scratch-to-Scale Project"

__version__ = "1.0.0"
__author__ = "AI-From-Scratch-to-Scale Project"
