# -*- coding: utf-8 -*-
"""The Perceptron model class.

This module defines the Perceptron class, which encapsulates the architecture and
learning algorithm of the Perceptron model. The implementation uses NumPy for
efficient numerical operations.

"""

import logging
from typing import Optional, List
import numpy as np
import time

# Handle both relative and absolute imports
try:
    from .constants import (
        DEFAULT_LEARNING_RATE,
        DEFAULT_ITERATIONS,
        DEFAULT_RANDOM_SEED
    )
except ImportError:
    from constants import (
        DEFAULT_LEARNING_RATE,
        DEFAULT_ITERATIONS,
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

    def __init__(
        self,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        n_iters: int = DEFAULT_ITERATIONS,
        logger: Optional[logging.Logger] = NULL_LOGGER,
        random_seed: Optional[int] = DEFAULT_RANDOM_SEED
    ) -> None:
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
        self.weights_history: List[np.ndarray] = []  # Store weights per epoch
        self.bias_history: List[float] = []  # Store bias per epoch
        self.accuracy_history: List[float] = []  # Store accuracy per epoch

        # Set random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        assert self.logger is not None  # type: ignore
        self.logger.info(
            f"Perceptron instance created. LR: {self.learning_rate}, "
            f"Iterations: {self.n_iters}"
        )

    def _heaviside_step_function(self, x: np.ndarray) -> np.ndarray:
        """Computes the Heaviside step function."""
        return np.where(x >= 0, 1, 0)

    def fit(self, features: np.ndarray, y: np.ndarray) -> None:
        """Trains the Perceptron model on the given dataset.

        The fitting process involves iterating over the dataset for `n_iters`
        epochs. In each epoch, the model updates its weights and bias for each
        misclassified sample.

        Args:
            X: The training input samples of shape (n_samples, n_features).
            y: The target values (class labels) of shape (n_samples,).
               These are expected to be binary (0 or 1).

        Raises:
            ValueError: If input shapes are invalid or incompatible.
        """
        # Input validation
        if not isinstance(features, np.ndarray):
            raise ValueError(f"features must be a numpy array, got {type(features)}")
        if not isinstance(y, np.ndarray):
            raise ValueError(f"y must be a numpy array, got {type(y)}")
        if features.ndim != 2:
            raise ValueError(f"features must be 2D array, got {features.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D")
        if features.shape[0] != y.shape[0]:
            raise ValueError(
                f"features and y must have same number of samples: "
                f"{features.shape[0]} vs {y.shape[0]}"
            )
        if features.shape[0] == 0:
            raise ValueError("Cannot fit on empty dataset")

        assert self.logger is not None  # type: ignore
        self.logger.info(f"Starting to fit the model on {features.shape[0]} samples.")
        _, n_features = features.shape

        # Initialize weights with proper dtype
        self.weights = np.random.rand(n_features).astype(np.float32) * 0.01
        self.bias = 0.0
        self.errors_per_epoch = []
        self.weights_history = []  # Reset history at start of fit
        self.bias_history = []  # Reset bias history
        self.accuracy_history = []  # Reset accuracy history

        # Ensure target variable is in {0, 1} format for robustness
        y_ = np.array([1 if i > 0 else 0 for i in y])

        import time
        for i in range(self.n_iters):
            epoch_start = time.perf_counter()
            errors_this_epoch = 0
            for idx, x_i in enumerate(features):
                assert self.weights is not None  # type: ignore
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._heaviside_step_function(linear_output)

                update = self.learning_rate * (y_[idx] - y_predicted)

                if update != 0:
                    self.weights += update * x_i
                    self.bias += update
                    errors_this_epoch += 1

            self.errors_per_epoch.append(errors_this_epoch)
            # Save a copy of the weights and bias after each epoch
            assert self.weights is not None  # type: ignore
            self.weights_history.append(self.weights.copy())
            self.bias_history.append(self.bias)
            # Calculate accuracy at the end of the epoch
            predictions = self.predict(features)
            accuracy = (predictions == y_).mean()
            self.accuracy_history.append(accuracy)

            # Compute precision, recall, f1_score
            tp = ((predictions == 1) & (y_ == 1)).sum()
            tn = ((predictions == 0) & (y_ == 0)).sum()
            fp = ((predictions == 1) & (y_ == 0)).sum()
            fn = ((predictions == 0) & (y_ == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            # Simulate probabilities for ROC-AUC (use linear_output normalized)
            from sklearn.metrics import roc_auc_score
            try:
                assert self.weights is not None  # type: ignore
                linear_outputs = np.dot(features, self.weights) + self.bias
                probs = (linear_outputs - linear_outputs.min()) / (linear_outputs.max() - linear_outputs.min() + 1e-8)
                roc_auc = roc_auc_score(y_, probs)
            except Exception:
                roc_auc = 0.0

            # Weight norm (L2)
            assert self.weights is not None  # type: ignore
            weight_norm = float(np.linalg.norm(self.weights))
            # Margin (min distance from decision boundary)
            try:
                assert self.weights is not None  # type: ignore
                margins = np.abs(np.dot(features, self.weights) + self.bias) / (np.linalg.norm(self.weights) + 1e-8)
                min_margin = float(np.min(margins))
            except Exception:
                min_margin = 0.0
            # Time per epoch
            epoch_time = time.perf_counter() - epoch_start

            # Log to W&B if active
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1_score,
                        "errors_per_epoch": errors_this_epoch,
                        "learning_rate": self.learning_rate,
                        "roc_auc": roc_auc,
                        "weight_norm": weight_norm,
                        "min_margin": min_margin,
                        "epoch_time": epoch_time
                    }, step=i+1)
            except ImportError:
                pass
            except Exception:
                pass

            # Log progress at a debug level to avoid cluttering the main console
            assert self.logger is not None  # type: ignore
            self.logger.debug(
                f"Epoch {i+1}/{self.n_iters} completed. "
                f"Updates: {errors_this_epoch}, Accuracy: {accuracy:.4f}"
            )

        assert self.logger is not None  # type: ignore
        self.logger.info("Fitting complete.")

    def predict(self, features: np.ndarray) -> np.ndarray:
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
        if not isinstance(features, np.ndarray):
            raise ValueError(f"features must be a numpy array, got {type(features)}")
        if features.ndim != 2:
            raise ValueError(f"features must be 2D array, got {features.ndim}D")
        if features.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Input features ({features.shape[1]}) don't match "
                f"trained features ({self.weights.shape[0]})"
            )

        assert self.logger is not None  # type: ignore
        self.logger.debug(f"Predicting on {features.shape[0]} samples.")
        assert self.weights is not None  # type: ignore
        linear_output = np.dot(features, self.weights) + self.bias
        y_predicted = self._heaviside_step_function(linear_output)
        return y_predicted

    def __repr__(self) -> str:
        """String representation of the Perceptron model.

        Returns:
            A string describing the model configuration and training state.
        """
        fitted_status = "fitted" if self.weights is not None else "not fitted"
        return (
            f"Perceptron(learning_rate={self.learning_rate}, "
            f"n_iters={self.n_iters}, {fitted_status})"
        )
