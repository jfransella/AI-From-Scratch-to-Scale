# -*- coding: utf-8 -*-
"""Multi-Layer Perceptron (MLP) implementation package.

This package provides a from-scratch implementation of a single-hidden-layer
neural network, demonstrating the fundamental concepts of deep learning:

- Forward propagation through linear layers and activation functions
- Backpropagation algorithm using the chain rule for gradient computation
- Xavier weight initialization for stable training
- Support for both binary and multi-class classification
- Comprehensive visualization of network behavior and learning dynamics
- Weights & Biases integration for experiment tracking

The implementation focuses on educational clarity, showing how modern
deep learning frameworks work under the hood through explicit mathematical
operations and detailed comments.

Example:
    >>> from src import MLP, MLPWandbVisualizer
    >>> model = MLP(input_size=784, hidden_size=128, output_size=10)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
"""

from .model import MLP
from .wandb_integration import MLPWandbVisualizer
from .config import EXPERIMENTS, WANDB_PROJECT_NAME

# Visualization functions - public API
from .visualize import (
    plot_confusion_matrix,
    plot_learning_curve,
    plot_decision_boundary,
    plot_neuron_weights
)

__all__ = [
    # Core model
    'MLP',
    
    # Experiment tracking
    'MLPWandbVisualizer',
    
    # Configuration
    'EXPERIMENTS',
    'WANDB_PROJECT_NAME',
    
    # Visualization functions
    'plot_confusion_matrix',
    'plot_learning_curve',
    'plot_decision_boundary',
    'plot_neuron_weights',
]

__version__ = "1.0.0"
__author__ = "AI-From-Scratch-to-Scale Project"
