# -*- coding: utf-8 -*-
"""The Perceptron model class.

This module defines the Perceptron class, which encapsulates the architecture and
learning algorithm of the Perceptron model. The implementation uses NumPy for
efficient numerical operations.

"""

import logging
from typing import Optional, List
import numpy as np
import wandb

# Handle both relative and absolute imports
try:
    from .constants import (
        DEFAULT_LEARNING_RATE, 
        DEFAULT_ITERATIONS, 
        MNIST_IMAGE_SIZE, 
        MNIST_WIDTH, 
        MNIST_HEIGHT,
        PIXEL_NORMALIZATION_FACTOR,
        DEFAULT_RANDOM_SEED
    )
except ImportError:
    from constants import (
        DEFAULT_LEARNING_RATE, 
        DEFAULT_ITERATIONS, 
        MNIST_IMAGE_SIZE, 
        MNIST_WIDTH, 
        MNIST_HEIGHT,
        PIXEL_NORMALIZATION_FACTOR,
        DEFAULT_RANDOM_SEED
    )

# A basic null logger for when no logger is passed to the class
NULL_LOGGER = logging.getLogger('null')
NULL_LOGGER.addHandler(logging.NullHandler())


class Perceptron:
    """A single-layer Perceptron for binary classification.

    This implementation of the Perceptron algorithm is designed for binary
    classification tasks. It learns a linear decision boundary to separate
    two classes.

    Attributes:
        learning_rate (float): The step size for weight updates.
        n_iters (int): The number of passes over the training dataset.
        weights (np.ndarray): The learned weights after fitting the model.
        bias (float): The learned bias term after fitting the model.
        errors_per_epoch (list[int]): A list containing the number of
                                      misclassifications in each epoch.
    """

    def __init__(self, learning_rate: float = DEFAULT_LEARNING_RATE, n_iters: int = DEFAULT_ITERATIONS, logger: Optional[logging.Logger] = NULL_LOGGER, random_seed: Optional[int] = DEFAULT_RANDOM_SEED) -> None:
        """Initializes the Perceptron model.

        Args:
            learning_rate: The learning rate for weight updates.
            n_iters: The number of iterations over the training data.
            logger: An optional logger instance.
            random_seed: Random seed for reproducibility.
            
        Raises:
            ValueError: If learning_rate <= 0 or n_iters <= 0.
        """
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {learning_rate}")
        if n_iters <= 0:
            raise ValueError(f"Number of iterations must be positive, got {n_iters}")
            
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.errors_per_epoch: List[int] = []
        self.logger = logger
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.logger.info(
            f"Perceptron instance created. LR: {self.learning_rate}, Iterations: {self.n_iters}"
        )

    def _heaviside_step_function(self, x: np.ndarray) -> np.ndarray:
        """Computes the Heaviside step function."""
        return np.where(x >= 0, 1, 0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the Perceptron model on the given dataset.

        The fitting process involves iterating over the dataset for `n_iters`
        epochs. In each epoch, the model updates its weights and bias for each
        misclassified sample. It also logs training metrics (accuracy, updates)
        and parameter distributions to Weights & Biases at the end of each epoch
        if W&B is enabled.

        Args:
            X: The training input samples of shape (n_samples, n_features).
            y: The target values (class labels) of shape (n_samples,).
               These are expected to be binary (0 or 1).
               
        Raises:
            ValueError: If input shapes are invalid or incompatible.
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise ValueError(f"X must be a numpy array, got {type(X)}")
        if not isinstance(y, np.ndarray):
            raise ValueError(f"y must be a numpy array, got {type(y)}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty dataset")
            
        self.logger.info(f"Starting to fit the model on {X.shape[0]} samples.")
        n_samples, n_features = X.shape

        # Initialize weights with proper dtype
        self.weights = np.random.rand(n_features).astype(np.float32) * 0.01
        self.bias = 0.0
        self.errors_per_epoch = []

        # Ensure target variable is in {0, 1} format for robustness
        y_ = np.array([1 if i > 0 else 0 for i in y])

        for i in range(self.n_iters):
            errors_this_epoch = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._heaviside_step_function(linear_output)

                update = self.learning_rate * (y_[idx] - y_predicted)
                
                if update != 0:
                    self.weights += update * x_i
                    self.bias += update
                    errors_this_epoch += 1
            
            self.errors_per_epoch.append(errors_this_epoch)
            
            # Calculate accuracy at the end of the epoch
            predictions = self.predict(X)
            accuracy = (predictions == y_).mean()

            # Log progress at a debug level to avoid cluttering the main console
            self.logger.debug(f"Epoch {i+1}/{self.n_iters} completed. Updates: {errors_this_epoch}, Accuracy: {accuracy:.4f}")
            
            # --- Log metrics and visualizations to Weights & Biases if enabled ---
            if wandb.run is not None and not wandb.run.disabled:
                log_data = {
                    "Training/Accuracy": accuracy,
                    "Training/Updates": errors_this_epoch,
                    "Parameters/Weights_Dist": wandb.Histogram(self.weights),
                    "Parameters/Bias_Dist": wandb.Histogram(self.bias),
                }

                # For MNIST, visualize the weights as an image
                if n_features == MNIST_IMAGE_SIZE:
                    # Reshape weights to a 28x28 image
                    img_weights = self.weights.reshape(MNIST_WIDTH, MNIST_HEIGHT)
                    
                    # Normalize the weights to the [0, 255] range for proper image logging
                    min_val, max_val = img_weights.min(), img_weights.max()
                    if max_val > min_val: # Avoid division by zero if all weights are the same
                        img_weights_normalized = PIXEL_NORMALIZATION_FACTOR * (img_weights - min_val) / (max_val - min_val)
                    else:
                        img_weights_normalized = np.zeros_like(img_weights)
                    
                    log_data["Parameters/Weights_Image"] = wandb.Image(img_weights_normalized)

                wandb.log(log_data, step=i)

        self.logger.info("Fitting complete.")


    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for the given input data.

        Args:
            X: The input samples to predict of shape (n_samples, n_features).

        Returns:
            An array of predicted class labels (0 or 1) of shape (n_samples,).
            
        Raises:
            ValueError: If model is not fitted or input shape is invalid.
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        if not isinstance(X, np.ndarray):
            raise ValueError(f"X must be a numpy array, got {type(X)}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Input features ({X.shape[1]}) don't match trained features ({self.weights.shape[0]})")
            
        self.logger.debug(f"Predicting on {X.shape[0]} samples.")
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._heaviside_step_function(linear_output)
        return y_predicted

    def __repr__(self) -> str:
        """String representation of the Perceptron model.
        
        Returns:
            A string describing the model configuration and training state.
        """
        fitted_status = "fitted" if self.weights is not None else "not fitted"
        return (f"Perceptron(learning_rate={self.learning_rate}, "
                f"n_iters={self.n_iters}, {fitted_status})")