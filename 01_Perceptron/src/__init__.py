# -*- coding: utf-8 -*-
"""Perceptron implementation package.

This package provides a from-scratch implementation of the Perceptron algorithm,
the foundation of neural networks. It includes:

- Core Perceptron model with training and prediction capabilities
- Data loaders for various binary classification datasets
- Visualization functions for decision boundaries and learning curves
- Weights & Biases integration for experiment tracking
- Configuration management for hyperparameters and experiments

The implementation focuses on educational clarity, showing the mathematical
foundations of neural learning without framework abstractions.

Example:
    >>> from src import Perceptron, PerceptronWandbVisualizer, PerceptronVisualizer
    >>> model = Perceptron(learning_rate=0.01, n_iters=100)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> visualizer = PerceptronVisualizer()
    >>> visualizer.generate_all_visualizations(model, X_train, y_train, predictions)
"""

from .model import Perceptron
from .wandb_integration import PerceptronWandbVisualizer
from .config import EXPERIMENTS, WANDB_PROJECT_NAME

# Visualization class - public API
from .visualize import PerceptronVisualizer

__all__ = [
    # Core model
    'Perceptron',    
    # Experiment tracking
    'PerceptronWandbVisualizer',    
    # Configuration
    'EXPERIMENTS',
    'WANDB_PROJECT_NAME',    
    # Visualization class
    'PerceptronVisualizer',
]

__version__ = "1.0.0"
__author__ = "AI-From-Scratch-to-Scale Project"
