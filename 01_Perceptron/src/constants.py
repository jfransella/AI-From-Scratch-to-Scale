# -*- coding: utf-8 -*-
"""Constants for the Perceptron project.

This module centralizes all constant values used throughout the project,
including default hyperparameters, image dimensions, and other fixed values.
"""

# --- Model Hyperparameters ---
DEFAULT_LEARNING_RATE: float = 0.01
DEFAULT_ITERATIONS: int = 1000

# --- MNIST Dataset Constants ---
MNIST_IMAGE_SIZE: int = 784  # 28x28 flattened
MNIST_WIDTH: int = 28
MNIST_HEIGHT: int = 28
PIXEL_NORMALIZATION_FACTOR: float = 255.0

# --- Visualization Constants ---
DECISION_BOUNDARY_RESOLUTION: float = 0.02
DEFAULT_FIGURE_SIZE: tuple = (10, 6)
DEFAULT_DPI: int = 300

# --- Random Seeds ---
DEFAULT_RANDOM_SEED: int = 42

# --- File Paths ---
DATA_DIR: str = "data"
OUTPUT_DIR: str = "outputs"
LOGS_DIR: str = "outputs/logs"
VISUALIZATIONS_DIR: str = "outputs/visualizations"
