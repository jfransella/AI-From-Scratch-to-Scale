# -*- coding: utf-8 -*-
"""Configuration settings for the Perceptron model.

This file centralizes all hyperparameters, file paths, and other
configuration values for the Perceptron project. Storing configuration
in a separate file makes it easier to manage and modify experimental
parameters without changing the core logic of the application.

"""

# --- Configuration for Logic Gate Experiments (AND, XOR) ---
LOGIC_GATE_LEARNING_RATE = 0.1
LOGIC_GATE_EPOCHS = 50

# A dictionary to hold paths to different logic gate datasets
LOGIC_GATE_DATA_PATHS = {
    "and": "data/perceptron_data.csv",
    "xor": "data/xor_data.csv",
}


# --- Configuration for MNIST Experiment ---
# MNIST is a more complex dataset and may require different settings.
# A single epoch over MNIST is much longer, so we start with fewer epochs.
MNIST_LEARNING_RATE = 0.01
MNIST_EPOCHS = 10