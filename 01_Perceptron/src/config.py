# -*- coding: utf-8 -*-
"""Configuration settings for the Perceptron model.

This file centralizes all hyperparameters, file paths, and other
configuration values for the Perceptron project. Storing configuration
in a separate file makes it easier to manage and modify experimental
parameters without changing the core logic of the application.

"""

from typing import Dict, Any

# Handle both relative and absolute imports
try:
    from .data_loader import load_perceptron_data, load_mnist_data, load_iris_data
except ImportError:
    from data_loader import load_perceptron_data, load_mnist_data, load_iris_data

# --- W&B Config ---
WANDB_PROJECT_NAME = "perceptron-from-scratch"

# --- Directory Constants ---
PLOTS_DIR: str = "outputs/plots"

# --- Experiment Registry ---
# Central place to define all parameters for each experiment.
# This makes adding new experiments much cleaner.

EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "and": {
        # The AND gate is a very simple, linearly separable problem.
        # A high learning rate and few epochs are sufficient for convergence.
        "data_loader": lambda: load_perceptron_data("data/perceptron_data.csv"),
        "learning_rate": 0.1,
        "epochs": 10,
        "class_names": ['False', 'True'],
    },
    "xor": {
        # The XOR gate is the classic example of a non-linearly separable problem.
        # The Perceptron will fail to converge. More epochs are set to demonstrate this.
        "data_loader": lambda: load_perceptron_data("data/xor_data.csv"),
        "learning_rate": 0.1,
        "epochs": 100,
        "class_names": ['False', 'True'],
    },
    "mnist": {
        # Classifying digits is a more complex task with higher dimensionality (784 features).
        # A smaller learning rate is a safer starting point for stable learning.
        "data_loader": load_mnist_data,
        "learning_rate": 0.01,
        "epochs": 10,
        "class_names": ['Digit 0', 'Digit 1'],
    },
    "iris-easy": {
        # Setosa vs. Versicolour is a linearly separable subset of the Iris dataset.
        # The Perceptron should converge easily.
        "data_loader": lambda: load_iris_data(class_indices=[0, 1]),
        "learning_rate": 0.01,
        "epochs": 100,
        "class_names": ['Setosa', 'Versicolour'],
    },
    "iris-hard": {
        # Versicolour vs. Virginica is a non-linearly separable subset.
        # The Perceptron will struggle to find a perfect boundary, similar to XOR.
        "data_loader": lambda: load_iris_data(class_indices=[1, 2]),
        "learning_rate": 0.01,
        "epochs": 100,
        "class_names": ['Versicolour', 'Virginica'],
    },
}
