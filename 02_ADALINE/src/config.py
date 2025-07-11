"""
Configuration settings for ADALINE (Adaptive Linear Neuron) implementation.

This module contains all hyperparameters, constants, and configuration settings
for the ADALINE model, following the project's centralized configuration pattern.
"""

import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class ADALINEConfig:
    """Configuration class for ADALINE model parameters and training settings."""
    
    # Model Architecture
    INPUT_SIZE: int = 2
    OUTPUT_SIZE: int = 1
    
    # Training Parameters
    LEARNING_RATE: float = 0.01
    MAX_EPOCHS: int = 1000
    BATCH_SIZE: int = 1  # ADALINE typically uses online learning
    CONVERGENCE_THRESHOLD: float = 1e-6
    MAX_ITERATIONS_WITHOUT_IMPROVEMENT: int = 50
    
    # Data Parameters
    RANDOM_SEED: int = 42
    TRAIN_TEST_SPLIT: float = 0.8
    VALIDATION_SPLIT: float = 0.2
    
    # Preprocessing
    NORMALIZE_FEATURES: bool = True
    ADD_BIAS_TERM: bool = True
    
    # Evaluation
    EVALUATION_METRICS: Tuple[str, ...] = ('mse', 'mae', 'r2_score')
    
    # Visualization
    PLOT_TRAINING_PROGRESS: bool = True
    PLOT_DECISION_BOUNDARY: bool = True
    PLOT_WEIGHT_EVOLUTION: bool = True
    SAVE_PLOTS: bool = True
    PLOT_DPI: int = 300
    
    # Logging
    LOG_LEVEL: str = 'INFO'
    LOG_TRAINING_PROGRESS: bool = True
    LOG_EVERY_N_EPOCHS: int = 10
    
    # File Paths
    OUTPUT_DIR: str = 'outputs'
    MODEL_SAVE_PATH: str = 'outputs/adaline_model.npz'
    PLOTS_SAVE_PATH: str = 'outputs/plots'
    LOGS_SAVE_PATH: str = 'outputs/logs'
    
    # Experiment Tracking
    USE_WANDB: bool = False
    WANDB_PROJECT_NAME: str = 'ai-from-scratch-adaline'
    WANDB_RUN_NAME: str = 'adaline-experiment'
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.LEARNING_RATE <= 0:
            raise ValueError("Learning rate must be positive")
        if self.MAX_EPOCHS <= 0:
            raise ValueError("Max epochs must be positive")
        if self.CONVERGENCE_THRESHOLD <= 0:
            raise ValueError("Convergence threshold must be positive")
        if not 0 < self.TRAIN_TEST_SPLIT < 1:
            raise ValueError("Train-test split must be between 0 and 1")
        if not 0 < self.VALIDATION_SPLIT < 1:
            raise ValueError("Validation split must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging and serialization."""
        return {
            'input_size': self.INPUT_SIZE,
            'output_size': self.OUTPUT_SIZE,
            'learning_rate': self.LEARNING_RATE,
            'max_epochs': self.MAX_EPOCHS,
            'batch_size': self.BATCH_SIZE,
            'convergence_threshold': self.CONVERGENCE_THRESHOLD,
            'max_iterations_without_improvement': self.MAX_ITERATIONS_WITHOUT_IMPROVEMENT,
            'random_seed': self.RANDOM_SEED,
            'train_test_split': self.TRAIN_TEST_SPLIT,
            'validation_split': self.VALIDATION_SPLIT,
            'normalize_features': self.NORMALIZE_FEATURES,
            'add_bias_term': self.ADD_BIAS_TERM,
            'evaluation_metrics': self.EVALUATION_METRICS,
            'plot_training_progress': self.PLOT_TRAINING_PROGRESS,
            'plot_decision_boundary': self.PLOT_DECISION_BOUNDARY,
            'plot_weight_evolution': self.PLOT_WEIGHT_EVOLUTION,
            'save_plots': self.SAVE_PLOTS,
            'plot_dpi': self.PLOT_DPI,
            'log_level': self.LOG_LEVEL,
            'log_training_progress': self.LOG_TRAINING_PROGRESS,
            'log_every_n_epochs': self.LOG_EVERY_N_EPOCHS,
            'output_dir': self.OUTPUT_DIR,
            'model_save_path': self.MODEL_SAVE_PATH,
            'plots_save_path': self.PLOTS_SAVE_PATH,
            'logs_save_path': self.LOGS_SAVE_PATH,
            'use_wandb': self.USE_WANDB,
            'wandb_project_name': self.WANDB_PROJECT_NAME,
            'wandb_run_name': self.WANDB_RUN_NAME,
        }


# Default configuration instance
config = ADALINEConfig()


# Constants for mathematical operations
EPSILON = 1e-8  # Small value to prevent division by zero
MAX_WEIGHT_VALUE = 10.0  # Maximum weight value for initialization
MIN_WEIGHT_VALUE = -10.0  # Minimum weight value for initialization

# Activation function constants
SIGMOID_STEEPNESS = 1.0  # Steepness parameter for sigmoid activation

# Convergence constants
MIN_IMPROVEMENT = 1e-8  # Minimum improvement for convergence
PATIENCE_FACTOR = 0.1  # Factor for early stopping patience

# Visualization constants
FIGURE_SIZE = (12, 8)
SUBPLOT_SIZE = (4, 3)
COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
STYLE_SETTINGS = {
    'figure.figsize': FIGURE_SIZE,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
}

# Error messages
ERROR_MESSAGES = {
    'invalid_input_shape': "Input shape {} doesn't match expected shape {}",
    'invalid_target_shape': "Target shape {} doesn't match expected shape {}",
    'convergence_failed': "Training failed to converge after {} epochs",
    'invalid_learning_rate': "Learning rate must be positive, got {}",
    'invalid_epochs': "Number of epochs must be positive, got {}",
    'data_empty': "Input data cannot be empty",
    'dimension_mismatch': "Input dimension {} doesn't match weight dimension {}",
    'invalid_activation': "Invalid activation function: {}",
    'model_not_fitted': "Model must be fitted before making predictions",
    'invalid_metric': "Invalid evaluation metric: {}",
}

# Success messages
SUCCESS_MESSAGES = {
    'training_converged': "Training converged after {} epochs",
    'model_saved': "Model saved to {}",
    'plots_saved': "Plots saved to {}",
    'evaluation_complete': "Evaluation completed successfully",
    'wandb_logged': "Experiment logged to Weights & Biases",
}

# Logging format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# File extensions
MODEL_EXTENSION = '.npz'
PLOT_EXTENSIONS = ['.png', '.pdf', '.svg']
LOG_EXTENSION = '.log'

# Performance thresholds
MAX_TRAINING_TIME = 300  # Maximum training time in seconds
MIN_ACCURACY_THRESHOLD = 0.8  # Minimum accuracy for successful training
MAX_LOSS_VALUE = 100.0  # Maximum acceptable loss value

# Mathematical constants for ADALINE
ADALINE_CONSTANTS = {
    'min_weight': -5.0,
    'max_weight': 5.0,
    'weight_scale': 0.1,
    'bias_scale': 0.01,
    'gradient_clip_threshold': 1.0,
    'momentum_factor': 0.9,
    'weight_decay': 1e-4,
} 