# -*- coding: utf-8 -*-
"""The Perceptron model class.

This module defines the Perceptron class, which encapsulates the architecture and
learning algorithm of the Perceptron model. The implementation uses NumPy for
efficient numerical operations.

"""

import logging
import numpy as np

# A basic null logger for when no logger is passed to the class
NULL_LOGGER = logging.getLogger('null')
NULL_LOGGER.addHandler(logging.NullHandler())


class Perceptron:
    """A single-layer Perceptron for binary classification.
    
    ... (docstring content is the same) ...
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
        """Trains the Perceptron model on the given dataset."""
        self.logger.info(f"Starting to fit the model on {X.shape[0]} samples.")
        n_samples, n_features = X.shape

        self.weights = np.random.rand(n_features) * 0.01
        self.bias = 0.0
        self.errors_per_epoch = []

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
            # Log progress at a debug level to avoid cluttering the main console
            self.logger.debug(f"Epoch {i+1}/{self.n_iters} completed. Updates in this epoch: {errors_this_epoch}")
        
        self.logger.info("Fitting complete.")


    def predict(self, X):
        """Predicts the class label for the given input data."""
        self.logger.info(f"Predicting on {X.shape[0]} samples.")
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._heaviside_step_function(linear_output)
        return y_predicted