# -*- coding: utf-8 -*-
"""Configuration settings for the MLP model.

This module centralizes all configuration values following the "No Hardcoded Values" principle.
All hyperparameters, paths, and experiment settings are defined here for easy experimentation
and reproducibility.

Educational Context:
    This configuration-driven approach demonstrates professional ML development practices
    where experiments can be easily modified without changing core implementation code.
"""

from typing import Dict, Any, Callable, List
from data_loader import load_logic_gate_data, load_mnist_multiclass_data, load_mnist_failure_test_data

# --- Weights & Biases Configuration ---
WANDB_PROJECT_NAME: str = "mlp-from-scratch"

# --- Training Constants ---
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_LEARNING_RATE: float = 0.01
DEFAULT_EPOCHS: int = 10000

# --- Data Constants ---
MNIST_IMAGE_SIZE: int = 784  # 28x28 flattened
MNIST_NUM_CLASSES: int = 10
XOR_INPUT_SIZE: int = 2
XOR_OUTPUT_SIZE: int = 1

# --- Visualization Constants ---
MAX_IMAGES_TO_LOG: int = 16
MAX_NEURONS_TO_VISUALIZE: int = 16
PLOT_RESOLUTION: float = 0.02
FIGURE_DPI: int = 300
PLOTS_DIR: str = "outputs/plots"

# --- Experiment Registry ---
# Each experiment defines a complete configuration for training and evaluation
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "xor": {
        # Educational Context: The XOR gate is the classic non-linearly separable problem
        # that demonstrates why neural networks need hidden layers. A single perceptron
        # cannot solve XOR, but an MLP with one hidden layer can.
        "data_loader": lambda: load_logic_gate_data("data/xor_data.csv"),
        "input_size": XOR_INPUT_SIZE,
        "hidden_size": 4,  # Minimum size to solve XOR: 2 neurons can create 2 linear boundaries
        "output_size": XOR_OUTPUT_SIZE,
        "learning_rate": 0.1,  # Higher learning rate for simple problem
        "epochs": DEFAULT_EPOCHS,
        "class_names": ['False', 'True'],
        "description": "Classic XOR gate problem demonstrating non-linear separability"
    },
    "mnist-multiclass": {
        # Educational Context: MNIST is the "hello world" of computer vision
        # This multi-class problem requires softmax output and cross-entropy loss
        "data_loader": load_mnist_multiclass_data,
        "input_size": MNIST_IMAGE_SIZE,
        "hidden_size": 128,  # Empirically good balance between capacity and efficiency
        "output_size": MNIST_NUM_CLASSES,
        "learning_rate": DEFAULT_LEARNING_RATE,  # Lower learning rate for SGD on larger dataset
        "epochs": 20,  # Fewer epochs due to larger dataset size
        "class_names": [f'Digit {i}' for i in range(MNIST_NUM_CLASSES)],
        "description": "Full MNIST digit classification (10 classes)"
    },
    "mnist-failure-test": {
        # Educational Context: Robustness testing is crucial for real-world deployment
        # This experiment evaluates how well the model generalizes to shifted images,
        # simulating real-world variations in input data
        "data_loader": load_mnist_failure_test_data,
        "input_size": MNIST_IMAGE_SIZE,
        "hidden_size": 128,
        "output_size": MNIST_NUM_CLASSES,
        "learning_rate": DEFAULT_LEARNING_RATE,  # Not used when loading model
        "epochs": 20,  # Not used when loading model
        "class_names": [f'Digit {i}' for i in range(MNIST_NUM_CLASSES)],
        "description": "MNIST robustness test with randomly shifted images"
    },
}