# -*- coding: utf-8 -*-
"""The Perceptron model class.

This module defines the Perceptron class, which encapsulates the architecture and
learning algorithm of the Perceptron model. The implementation uses NumPy for
efficient numerical operations.

"""

import logging
import numpy as np
import wandb

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

    def __init__(self, learning_rate=0.01, n_iters=1000, logger=NULL_LOGGER):
        """Initializes the Perceptron model.

        Args:
            learning_rate (float): The learning rate for weight updates.
            n_iters (int): The number of iterations over the training data.
            logger (logging.Logger): An optional logger instance.
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []
        self.logger = logger
        self.logger.info(
            f"Perceptron instance created. LR: {self.learning_rate}, Iterations: {self.n_iters}"
        )

    def _heaviside_step_function(self, x):
        """Computes the Heaviside step function."""
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """Trains the Perceptron model on the given dataset.

        The fitting process involves iterating over the dataset for `n_iters`
        epochs. In each epoch, the model updates its weights and bias for each
        misclassified sample. It also logs training metrics (accuracy, updates)
        and parameter distributions to Weights & Biases at the end of each epoch
        if W&B is enabled.

        Args:
            X (np.ndarray): The training input samples.
            y (np.ndarray): The target values (class labels). Shape (n_samples,).
                            These are expected to be binary (0 or 1).

        Returns:
            None
        """
        self.logger.info(f"Starting to fit the model on {X.shape[0]} samples.")
        n_samples, n_features = X.shape

        self.weights = np.random.rand(n_features) * 0.01
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
            if not wandb.run.disabled:
                log_data = {
                    "Training/Accuracy": accuracy,
                    "Training/Updates": errors_this_epoch,
                    "Parameters/Weights_Dist": wandb.Histogram(self.weights),
                    "Parameters/Bias_Dist": wandb.Histogram(self.bias),
                }

                # For MNIST, visualize the weights as an image
                if n_features == 784:
                    # Reshape weights to a 28x28 image
                    img_weights = self.weights.reshape(28, 28)
                    
                    # Normalize the weights to the [0, 255] range for proper image logging
                    min_val, max_val = img_weights.min(), img_weights.max()
                    if max_val > min_val: # Avoid division by zero if all weights are the same
                        img_weights_normalized = 255 * (img_weights - min_val) / (max_val - min_val)
                    else:
                        img_weights_normalized = np.zeros_like(img_weights)
                    
                    log_data["Parameters/Weights_Image"] = wandb.Image(img_weights_normalized)

                wandb.log(log_data, step=i)

        self.logger.info("Fitting complete.")


    def predict(self, X):
        """Predicts class labels for the given input data.

        Args:
            X (np.ndarray): The input samples to predict. Shape (n_samples, n_features).

        Returns:
            np.ndarray: An array of predicted class labels (0 or 1).
        """
        self.logger.debug(f"Predicting on {X.shape[0]} samples.")
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._heaviside_step_function(linear_output)
        return y_predicted