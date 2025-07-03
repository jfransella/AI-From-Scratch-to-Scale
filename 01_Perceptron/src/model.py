# -*- coding: utf-8 -*-
"""The Perceptron model class.

This module defines the Perceptron class, which encapsulates the architecture and
learning algorithm of the Perceptron model. The implementation uses NumPy for
efficient numerical operations.

"""

import numpy as np


class Perceptron:
    """A single-layer Perceptron for binary classification.

    The Perceptron is one of the simplest types of artificial neural networks.
    It is a model of a single neuron that can be used for two-class
    classification problems.

    Attributes:
        learning_rate (float): The step size for weight updates.
        n_iters (int): The number of passes over the training dataset.
        weights (np.ndarray): The weights applied to the input features.
        bias (float): The bias term.

    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        """Initializes the Perceptron model.

        Args:
            learning_rate (float): The learning rate for weight updates.
            n_iters (int): The number of iterations over the training data.
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _heaviside_step_function(self, x):
        """Computes the Heaviside step function.

        This activation function returns 1 if the input is non-negative,
        and 0 otherwise. It is the decision-making unit of the Perceptron.

        Args:
            x (np.ndarray): The linear combination of weights, inputs, and bias.

        Returns:
            np.ndarray: The binary output (0 or 1).
        """
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """Trains the Perceptron model on the given dataset.

        The method initializes weights and bias, then iterates over the data
        for `n_iters` epochs, updating the weights and bias based on the
        Perceptron learning rule each time a misclassification occurs.

        Args:
            X (np.ndarray): The training input samples. Shape (n_samples, n_features).
            y (np.ndarray): The target values (true labels). Shape (n_samples,).
        """
        n_samples, n_features = X.shape

        # Initialize weights and bias.
        # We use small random numbers instead of zeros to break symmetry.
        self.weights = np.random.rand(n_features) * 0.01
        self.bias = 0.0

        # Ensure y is in the correct format (0s and 1s)
        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._heaviside_step_function(linear_output)

                # Perceptron update rule
                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """Predicts the class label for the given input data.

        Args:
            X (np.ndarray): The input samples for which to make predictions.

        Returns:
            np.ndarray: The predicted class labels (0 or 1).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._heaviside_step_function(linear_output)
        return y_predicted