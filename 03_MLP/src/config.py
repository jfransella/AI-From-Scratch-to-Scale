# -*- coding: utf-8 -*-
"""Configuration settings for the MLP model."""

from typing import Dict, Any, Callable, List
from src.data_loader import load_logic_gate_data, load_mnist_multiclass_data, load_mnist_failure_test_data

# --- W&B Config ---
WANDB_PROJECT_NAME: str = "mlp-from-scratch"

# --- Experiment Registry ---
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "xor": {
        # The XOR gate is the classic non-linearly separable problem.
        # An MLP with a hidden layer is required to solve it.
        "data_loader": lambda: load_logic_gate_data("data/xor_data.csv"),
        "input_size": 2,
        "hidden_size": 4,  # A good starting point for a simple problem
        "output_size": 1,
        "learning_rate": 0.1,
        "epochs": 10000,
        "class_names": ['False', 'True'],
    },
    "mnist-multiclass": {
        # The full MNIST dataset with 10 classes (digits 0-9).
        # This requires a multi-class output layer (Softmax) and loss function.
        "data_loader": load_mnist_multiclass_data,
        "input_size": 784,
        "hidden_size": 128,  # A larger hidden layer for a more complex task
        "output_size": 10,   # 10 output neurons, one for each digit
        "learning_rate": 0.01,  # SGD requires a smaller learning_rate
        "epochs": 20,
        "class_names": [f'Digit {i}' for i in range(10)],
    },
    "mnist-failure-test": {
        # The MNIST dataset, but the test set images are randomly shifted.
        # This is used to evaluate a model's robustness to translation.
        "data_loader": load_mnist_failure_test_data,
        "input_size": 784,
        "hidden_size": 128,
        "output_size": 10,
        "learning_rate": 0.01,  # Not used when loading a model, but good to have
        "epochs": 20,          # Not used when loading a model
        "class_names": [f'Digit {i}' for i in range(10)],
    },
}