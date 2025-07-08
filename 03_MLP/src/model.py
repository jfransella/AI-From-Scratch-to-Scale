# -*- coding: utf-8 -*-
"""The Multi-Layer Perceptron (MLP) model class."""

import logging
import numpy as np
import wandb

# A basic null logger for when no logger is passed to the class
NULL_LOGGER = logging.getLogger('null')
NULL_LOGGER.addHandler(logging.NullHandler())

class MLP:
    """A Multi-Layer Perceptron for binary classification with one hidden layer.

    Attributes:
        input_size (int): The number of input features.
        hidden_size (int): The number of neurons in the hidden layer.
        output_size (int): The number of output neurons.
        learning_rate (float): The step size for weight updates.
        epochs (int): The number of passes over the training dataset.
        W1 (np.ndarray): Weights for the input to hidden layer.
        b1 (np.ndarray): Biases for the hidden layer.
        W2 (np.ndarray): Weights for the hidden to output layer.
        b2 (np.ndarray): Biases for the output layer.
        losses (list[float]): A list of the loss for each epoch.
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10000, logger=NULL_LOGGER, wandb_run=None):
        """Initializes the MLP model.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of output neurons.
            learning_rate (float): The learning rate for weight updates.
            epochs (int): The number of iterations over the training data.
            logger (logging.Logger): An optional logger instance.
            wandb_run: The active wandb run object for logging.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.logger = logger
        self.wandb_run = wandb_run

        # Initialize weights with small random values to break symmetry
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

        # To store loss history for plotting
        self.losses = []

    def _sigmoid(self, x):
        """A numerically stable Sigmoid activation function."""
        # Prevent overflow by clipping large values
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        """The Softmax activation function for the output layer."""
        # Subtract max for numerical stability, preventing overflow.
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _sigmoid_derivative(self, x):
        """The derivative of the Sigmoid function."""
        return x * (1 - x)

    def _forward(self, X):
        """Performs the forward pass through the network.

        Args:
            X (np.ndarray): The input data.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the hidden layer
                                           activations (a1) and the final
                                           output activations (a2).
        """
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._sigmoid(self.z1)

        # Hidden to output layer
        # For multi-class, we use Softmax in the output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        if self.output_size > 1:
            self.a2 = self._softmax(self.z2)
        else:  # Binary classification
            self.a2 = self._sigmoid(self.z2)
        return self.a1, self.a2

    def predict(self, X):
        """Predicts class labels for the given input data."""
        _, output = self._forward(X)
        if self.output_size > 1:
            # For multi-class, return the index of the highest probability
            return np.argmax(output, axis=1)
        else:  # Binary classification
            return (output > 0.5).astype(int)

    def save_model(self, filepath):
        """Saves the model's weights and biases to a file.

        Args:
            filepath (str): The path to the file where the model will be saved.
        """
        self.logger.info(f"Saving model to {filepath}...")
        np.savez(
            filepath,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
        )
        self.logger.info("Model saved successfully.")

    def load_model(self, filepath):
        """Loads the model's weights and biases from a file.

        Args:
            filepath (str): The path to the file from which to load the model.
        """
        self.logger.info(f"Loading model from {filepath}...")
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.logger.info("Model loaded successfully.")

    def fit(self, X, y):
        """Trains the MLP model using stochastic gradient descent.

        Args:
            X (np.ndarray): The training input samples.
            y (np.ndarray): The target values (class labels).
        """
        self.losses = []
        n_samples = X.shape[0]
        global_step = 0

        for i in range(self.epochs):
            # --- Stochastic Gradient Descent ---
            # Iterate over each training sample individually
            for j in range(n_samples):
                global_step += 1

                # Log progress every 10,000 samples to show the bottleneck
                if (j + 1) % 10000 == 0:
                    self.logger.info(f"  Epoch {i+1}/{self.epochs}, Sample {j+1}/{n_samples}")

                x_sample = X[j:j+1]  # Keep it as a 2D array (1, n_features)
                y_sample = y[j:j+1]  # Keep it as a 2D array (1, n_outputs)

                # --- Forward Pass ---
                a1, a2 = self._forward(x_sample)

                # --- Log Sample Loss Periodically ---
                if self.wandb_run and not self.wandb_run.run.disabled and (j + 1) % 100 == 0:
                    if self.output_size > 1:
                        # Use a stable way to get the probability of the true class
                        true_class_idx = np.argmax(y_sample, axis=1)
                        prob = a2[0, true_class_idx]
                        sample_loss = -np.log(prob + 1e-9)
                    else:
                        sample_loss = 0.5 * (y_sample - a2) ** 2

                    self.wandb_run.log(
                        {"Training/Sample Loss": sample_loss.item()}, step=global_step
                    )

                # --- Backward Pass (Backpropagation) ---
                if self.output_size > 1:
                    # Gradient for Softmax + Cross-Entropy is simply (predicted - true)
                    delta_output = a2 - y_sample
                else:
                    # Gradient for Sigmoid + MSE
                    delta_output = (a2 - y_sample) * self._sigmoid_derivative(a2)

                error_hidden = delta_output.dot(self.W2.T)
                delta_hidden = error_hidden * self._sigmoid_derivative(a1)

                # Calculate gradients for the weights and biases for this single sample
                d_W2 = a1.T.dot(delta_output)
                d_b2 = np.sum(delta_output, axis=0, keepdims=True)
                d_W1 = x_sample.T.dot(delta_hidden)
                d_b1 = np.sum(delta_hidden, axis=0, keepdims=True)

                # Update weights and biases immediately after each sample
                self.W1 -= self.learning_rate * d_W1
                self.b1 -= self.learning_rate * d_b1
                self.W2 -= self.learning_rate * d_W2
                self.b2 -= self.learning_rate * d_b2

            # --- Calculate and record loss for the entire dataset at the end of the epoch ---
            _, full_output = self._forward(X)
            if self.output_size > 1:
                log_probs = -np.log(full_output[np.arange(n_samples), np.argmax(y, axis=1)] + 1e-9)
                loss = np.sum(log_probs) / n_samples
            else:
                loss = np.mean(0.5 * (full_output - y) ** 2)
            self.losses.append(loss)
            
            if self.wandb_run and not self.wandb_run.run.disabled:
                self.wandb_run.log({"Training/Epoch Loss": loss}, step=global_step)