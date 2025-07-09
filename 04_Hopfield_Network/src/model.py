# -*- coding: utf-8 -*-
"""The Hopfield Network model class."""

import numpy as np


class HopfieldNetwork:
    """
    A Hopfield Network for storing and recalling patterns.

    This network acts as a form of associative memory. It is trained on a set
    of bipolar patterns (-1, 1) and can later reconstruct a full pattern from
    a noisy or incomplete version.
    """

    def __init__(self, num_neurons):
        """
        Initializes the network.

        Args:
            num_neurons (int): The number of neurons in the network, which
                               corresponds to the size of the patterns.
        """
        self.num_neurons = num_neurons
        # Initialize weights as a square matrix of zeros.
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        """
        Trains the network on a set of patterns using the Hebbian learning rule.

        Args:
            patterns (list of np.ndarray): A list of patterns to store. Each
                                           pattern should be a 1D NumPy array
                                           of bipolar values (-1, 1).
        """
        # The Hebbian rule: W_ij = sum(p_i * p_j) for all patterns p.
        for p in patterns:
            self.weights += np.outer(p, p)

        # Set the diagonal to zero to prevent self-connections.
        np.fill_diagonal(self.weights, 0)

    def predict(self, pattern, max_iter=20):
        """
        Recalls a pattern from a given (potentially noisy) input pattern.

        The network state is updated asynchronously until it converges to a
        stable state or the maximum number of iterations is reached.

        Args:
            pattern (np.ndarray): The input pattern to recall.
            max_iter (int): The maximum number of update iterations.

        Returns:
            np.ndarray: The recalled (stable) pattern.
        """
        current_state = np.copy(pattern)

        for _ in range(max_iter):
            # Asynchronous update: update one neuron at a time in a random order.
            for i in np.random.permutation(self.num_neurons):
                activation = np.dot(self.weights[i, :], current_state)
                current_state[i] = 1 if activation >= 0 else -1

        return current_state