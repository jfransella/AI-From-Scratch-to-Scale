# -*- coding: utf-8 -*-
"""The Multi-Layer Perceptron (MLP) model class."""

import logging
from typing import Tuple, Optional, List
import numpy as np
import wandb

logger = logging.getLogger(__name__)

class MLP:
    """A Multi-Layer Perceptron for binary and multi-class classification with one hidden layer.

    This implementation uses stochastic gradient descent for training and supports
    both binary classification (sigmoid + MSE) and multi-class classification 
    (softmax + cross-entropy).

    Attributes:
        input_size: The number of input features
        hidden_size: The number of neurons in the hidden layer
        output_size: The number of output neurons
        learning_rate: The step size for weight updates
        epochs: The number of passes over the training dataset
        W1: Weights for the input to hidden layer of shape (input_size, hidden_size)
        b1: Biases for the hidden layer of shape (1, hidden_size)
        W2: Weights for the hidden to output layer of shape (hidden_size, output_size)
        b2: Biases for the output layer of shape (1, output_size)
        losses: A list of the loss for each epoch
    """

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int, 
        learning_rate: float = 0.1, 
        epochs: int = 10000, 
        random_seed: Optional[int] = 42,
        wandb_run: Optional[object] = None
    ) -> None:
        """Initializes the MLP model.

        Args:
            input_size: The number of input features
            hidden_size: The number of neurons in the hidden layer
            output_size: The number of output neurons
            learning_rate: The learning rate for weight updates
            epochs: The number of iterations over the training data
            random_seed: Random seed for reproducibility. Set to None for no seeding
            wandb_run: The active wandb run object for logging
            
        Raises:
            ValueError: If any size parameter is <= 0 or learning_rate <= 0
        """
        # Validate inputs
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
            
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.wandb_run = wandb_run

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            logger.info(f"Set random seed to {random_seed}")

        # Initialize weights with small random values to break symmetry
        # Using Xavier initialization scaled for better convergence
        self.W1 = np.random.randn(self.input_size, self.hidden_size).astype(np.float32) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size), dtype=np.float32)
        self.W2 = np.random.randn(self.hidden_size, self.output_size).astype(np.float32) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size), dtype=np.float32)

        # To store loss history for plotting
        self.losses: List[float] = []
        
        logger.info(f"Initialized MLP: {input_size}->{hidden_size}->{output_size}, lr={learning_rate}")

    def __repr__(self) -> str:
        """Returns a string representation of the model."""
        return (f"MLP(input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"output_size={self.output_size}, learning_rate={self.learning_rate})")

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """A numerically stable Sigmoid activation function.
        
        Args:
            x: Input array
            
        Returns:
            Sigmoid activations
        """
        # Prevent overflow by clipping large values
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """The Softmax activation function for the output layer.
        
        Args:
            x: Input logits
            
        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability, preventing overflow
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """The derivative of the Sigmoid function.
        
        Args:
            x: Sigmoid activations (not logits)
            
        Returns:
            Sigmoid derivatives
        """
        return x * (1 - x)

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Performs the forward pass through the network.

        Args:
            X: The input data of shape (batch_size, input_size)

        Returns:
            A tuple containing the hidden layer activations (a1) and 
            the final output activations (a2)
            
        Raises:
            ValueError: If input dimensions don't match expected input_size
        """
        if X.shape[1] != self.input_size:
            raise ValueError(f"Input dimension {X.shape[1]} doesn't match expected {self.input_size}")
        
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for the given input data.
        
        Args:
            X: Input data of shape (n_samples, input_size)
            
        Returns:
            Predicted class labels
            
        Raises:
            ValueError: If input dimensions are incorrect
        """
        if X.ndim != 2:
            raise ValueError(f"Input must be 2D array, got {X.ndim}D")
        
        _, output = self._forward(X)
        if self.output_size > 1:
            # For multi-class, return the index of the highest probability
            return np.argmax(output, axis=1)
        else:  # Binary classification
            return (output > 0.5).astype(int).flatten()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score on given data.
        
        Args:
            X: Input features of shape (n_samples, input_size)
            y: True labels
            
        Returns:
            Accuracy score between 0 and 1
        """
        predictions = self.predict(X)
        
        # Handle one-hot encoded labels
        if y.ndim > 1 and y.shape[1] > 1:
            y_labels = np.argmax(y, axis=1)
        else:
            y_labels = y.flatten()
            
        return np.mean(predictions == y_labels)

    def save_model(self, filepath: str) -> None:
        """Saves the model's weights and biases to a file.

        Args:
            filepath: The path to the file where the model will be saved
            
        Raises:
            IOError: If the file cannot be written
        """
        try:
            logger.info(f"Saving model to {filepath}...")
            np.savez(
                filepath,
                W1=self.W1,
                b1=self.b1,
                W2=self.W2,
                b2=self.b2,
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                learning_rate=self.learning_rate
            )
            logger.info("Model saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save model to {filepath}: {e}")
            raise IOError(f"Failed to save model: {e}")

    def load_model(self, filepath: str) -> None:
        """Loads the model's weights and biases from a file.

        Args:
            filepath: The path to the file from which to load the model
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
            ValueError: If the model file is corrupted or incompatible
        """
        try:
            logger.info(f"Loading model from {filepath}...")
            data = np.load(filepath)
            
            # Validate loaded data
            required_keys = ['W1', 'b1', 'W2', 'b2']
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise ValueError(f"Model file missing required keys: {missing_keys}")
            
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            
            # Load metadata if available
            if 'input_size' in data:
                self.input_size = int(data['input_size'])
            if 'hidden_size' in data:
                self.hidden_size = int(data['hidden_size'])
            if 'output_size' in data:
                self.output_size = int(data['output_size'])
            if 'learning_rate' in data:
                self.learning_rate = float(data['learning_rate'])
                
            logger.info("Model loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
            raise ValueError(f"Failed to load model: {e}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the MLP model using stochastic gradient descent.

        Args:
            X: The training input samples of shape (n_samples, input_size)
            y: The target values (class labels) of shape (n_samples, output_size)
            
        Raises:
            ValueError: If input dimensions are incorrect
        """
        # Validate inputs
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if X.shape[1] != self.input_size:
            raise ValueError(f"X second dimension {X.shape[1]} doesn't match input_size {self.input_size}")
        
        logger.info(f"Starting training for {self.epochs} epochs on {X.shape[0]} samples")
        
        self.losses = []
        n_samples = X.shape[0]
        global_step = 0

        for i in range(self.epochs):
            # --- Stochastic Gradient Descent ---
            # Iterate over each training sample individually
            epoch_loss = 0.0
            
            for j in range(n_samples):
                global_step += 1

                # Log progress every 10,000 samples to show the bottleneck
                if (j + 1) % 10000 == 0:
                    logger.info(f"  Epoch {i+1}/{self.epochs}, Sample {j+1}/{n_samples}")

                x_sample = X[j:j+1]  # Keep it as a 2D array (1, n_features)
                y_sample = y[j:j+1]  # Keep it as a 2D array (1, n_outputs)

                # --- Forward Pass ---
                a1, a2 = self._forward(x_sample)

                # --- Calculate Sample Loss ---
                if self.output_size > 1:
                    # Cross-entropy loss for multi-class
                    true_class_idx = np.argmax(y_sample, axis=1)
                    prob = a2[0, true_class_idx[0]]
                    sample_loss = -np.log(prob + 1e-9)
                else:
                    # MSE loss for binary classification
                    sample_loss = 0.5 * np.mean((y_sample - a2) ** 2)

                epoch_loss += sample_loss

                # --- Log Sample Loss Periodically ---
                if (self.wandb_run and not self.wandb_run.run.disabled and 
                    (j + 1) % 100 == 0):
                    self.wandb_run.log(
                        {"Training/Sample Loss": float(sample_loss)}, step=global_step
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
            epoch_loss = epoch_loss / n_samples
            self.losses.append(float(epoch_loss))
            
            if (self.wandb_run and not self.wandb_run.run.disabled):
                self.wandb_run.log({"Training/Epoch Loss": float(epoch_loss)}, step=global_step)
                
            # Log epoch progress
            if (i + 1) % max(1, self.epochs // 10) == 0:
                logger.info(f"Epoch {i+1}/{self.epochs}, Loss: {epoch_loss:.6f}")
        
        logger.info(f"Training completed. Final loss: {self.losses[-1]:.6f}")