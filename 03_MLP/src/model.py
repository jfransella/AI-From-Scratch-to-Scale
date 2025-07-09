# -*- coding: utf-8 -*-
"""The Multi-Layer Perceptron (MLP) model class.

This module implements a single-hidden-layer neural network from scratch using only NumPy.
The implementation demonstrates fundamental concepts in deep learning including:
- Forward propagation through linear layers and activation functions
- Backpropagation algorithm using the chain rule
- Stochastic gradient descent optimization
- Support for both binary and multi-class classification

Educational Context:
    This MLP represents the foundational building block of deep learning.
    By implementing it from scratch, we understand how modern frameworks
    work under the hood and appreciate the mathematical elegance of
    backpropagation for automatic differentiation.

Mathematical Background:
    Forward Pass:
        z1 = X @ W1 + b1        # Linear transformation to hidden layer
        a1 = σ(z1)              # Hidden layer activations (σ = sigmoid)
        z2 = a1 @ W2 + b2       # Linear transformation to output
        a2 = σ(z2) or softmax(z2)  # Output activations
    
    Backward Pass (Chain Rule):
        ∂L/∂W2 = a1.T @ δ2      # Gradient w.r.t. output weights
        ∂L/∂W1 = X.T @ δ1       # Gradient w.r.t. hidden weights
        where δ represents error terms computed via chain rule
"""

import logging
from typing import Tuple, Optional, List
import numpy as np
import wandb

logger = logging.getLogger(__name__)

# Mathematical constants for numerical stability
SIGMOID_CLIP_VALUE: float = 500.0  # Prevents overflow in sigmoid
EPSILON: float = 1e-9  # Small value to prevent log(0) in cross-entropy

class MLP:
    """A Multi-Layer Perceptron for binary and multi-class classification with one hidden layer.
    
    Educational Context:
        This MLP demonstrates the fundamental concepts that led to the deep learning revolution:
        1. Universal Approximation: Any continuous function can be approximated by an MLP
        2. Non-linear Mapping: Hidden layers with activation functions enable non-linear decision boundaries
        3. Gradient-based Learning: Backpropagation enables efficient training of multi-layer networks
        4. Representation Learning: Hidden layers learn useful feature representations
    
    Architecture:
        Input Layer (n features) → Hidden Layer (m neurons, sigmoid) → Output Layer (k classes)
    
    This implementation uses stochastic gradient descent for training and supports
    both binary classification (sigmoid + MSE) and multi-class classification 
    (softmax + cross-entropy).

    Attributes:
        input_size: The number of input features
        hidden_size: The number of neurons in the hidden layer  
        output_size: The number of output neurons (1 for binary, k for k-class)
        learning_rate: The step size for weight updates (typically 0.001-0.1)
        epochs: The number of complete passes through the training dataset
        W1: Weights for input→hidden transformation of shape (input_size, hidden_size)
        b1: Biases for the hidden layer of shape (1, hidden_size)
        W2: Weights for hidden→output transformation of shape (hidden_size, output_size)
        b2: Biases for the output layer of shape (1, output_size)
        losses: A list storing the loss value for each training epoch
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
        """Initializes the MLP model with Xavier weight initialization.
        
        Educational Context:
            Proper weight initialization is crucial for neural network training:
            - Xavier/Glorot initialization scales weights based on layer sizes
            - This prevents vanishing/exploding gradients in deeper networks
            - Biases are typically initialized to zero
            - Random seeds ensure reproducible experiments

        Args:
            input_size: The number of input features (e.g., 784 for MNIST, 2 for XOR)
            hidden_size: The number of neurons in the hidden layer (typically 64-512)
            output_size: The number of output neurons (1 for binary, k for k-class)
            learning_rate: The learning rate for gradient descent (typically 0.001-0.1)
            epochs: The number of complete passes through the training data
            random_seed: Random seed for reproducibility. Set to None for no seeding
            wandb_run: The active wandb run object for experiment logging
            
        Raises:
            ValueError: If any size parameter is <= 0 or learning_rate <= 0
        """
        # Validate all hyperparameters
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
            
        # Store architecture and hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.wandb_run = wandb_run

        # Set random seed for reproducible experiments
        if random_seed is not None:
            np.random.seed(random_seed)
            logger.info(f"Set random seed to {random_seed} for reproducible initialization")

        # Initialize weights using Xavier/Glorot initialization
        # This scales weights based on the number of input connections
        # Formula: weight ~ N(0, sqrt(2/n_in)) for better gradient flow
        self.W1 = np.random.randn(self.input_size, self.hidden_size).astype(np.float32) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size), dtype=np.float32)  # Biases start at zero
        self.W2 = np.random.randn(self.hidden_size, self.output_size).astype(np.float32) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size), dtype=np.float32)

        # Initialize training history storage
        self.losses: List[float] = []
        
        logger.info(f"Initialized MLP: {input_size}→{hidden_size}→{output_size}, lr={learning_rate}")
        logger.info(f"Weight shapes: W1{self.W1.shape}, W2{self.W2.shape}")

    def __repr__(self) -> str:
        """Returns a string representation of the model for debugging."""
        return (f"MLP(input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"output_size={self.output_size}, learning_rate={self.learning_rate})")

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable Sigmoid activation function.
        
        Educational Context:
            The sigmoid function σ(x) = 1/(1 + e^(-x)) is a classic activation:
            - Maps any real number to the range (0, 1)
            - Smooth and differentiable everywhere
            - Historically important but can cause vanishing gradients
            - Still useful for binary classification output layers
        
        Mathematical Properties:
            - Range: (0, 1)
            - Derivative: σ'(x) = σ(x)(1 - σ(x))
            - Saturation: gradients → 0 as |x| → ∞
        
        Args:
            x: Input array of any shape
            
        Returns:
            Sigmoid activations with same shape as input
        """
        # Prevent overflow by clipping extreme values
        # For x > 500, exp(-x) ≈ 0, so sigmoid ≈ 1
        # For x < -500, exp(-x) → ∞, so sigmoid ≈ 0
        x_clipped = np.clip(x, -SIGMOID_CLIP_VALUE, SIGMOID_CLIP_VALUE)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable Softmax activation function for multi-class output.
        
        Educational Context:
            Softmax is the standard activation for multi-class classification:
            - Converts logits to probability distribution (sums to 1)
            - Emphasizes the largest input (argmax behavior)
            - Works naturally with cross-entropy loss
            - Generalizes sigmoid to multiple classes
        
        Mathematical Formula:
            softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
            
        Numerical Stability:
            We subtract max(x) to prevent overflow: exp(x - max(x))
        
        Args:
            x: Input logits of shape (batch_size, n_classes)
            
        Returns:
            Probability distribution of shape (batch_size, n_classes)
        """
        # Subtract max for numerical stability (prevents overflow)
        # This doesn't change the softmax output: softmax(x) = softmax(x - c)
        x_stable = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _sigmoid_derivative(self, sigmoid_output: np.ndarray) -> np.ndarray:
        """Derivative of the sigmoid function.
        
        Educational Context:
            The sigmoid derivative has a beautiful property:
            If a = σ(z), then σ'(z) = a(1-a)
            This means we can compute the derivative using only the sigmoid output,
            which is computationally efficient during backpropagation.
        
        Mathematical Derivation:
            σ(z) = 1/(1 + e^(-z))
            σ'(z) = σ(z)(1 - σ(z))
        
        Args:
            sigmoid_output: Already computed sigmoid activations (not raw logits!)
            
        Returns:
            Sigmoid derivatives with same shape as input
        """
        return sigmoid_output * (1.0 - sigmoid_output)

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Performs the forward pass through the network.
        
        Educational Context:
            Forward propagation computes the network's predictions by passing
            input data through each layer sequentially. This demonstrates:
            1. Linear transformations: z = Wx + b
            2. Non-linear activations: a = f(z)
            3. Information flow from input to output
            4. How representations are built layer by layer
        
        Mathematical Steps:
            1. Hidden layer: z1 = X @ W1 + b1, a1 = σ(z1)
            2. Output layer: z2 = a1 @ W2 + b2, a2 = σ(z2) or softmax(z2)
        
        Note: We store intermediate values (z1, z2, a1) as instance variables
        because they're needed during backpropagation.

        Args:
            X: Input data of shape (batch_size, input_size)

        Returns:
            A tuple containing:
            - a1: Hidden layer activations of shape (batch_size, hidden_size)
            - a2: Output activations of shape (batch_size, output_size)
            
        Raises:
            ValueError: If input dimensions don't match expected input_size
        """
        # Validate input dimensions for debugging
        if X.shape[1] != self.input_size:
            raise ValueError(
                f"Input dimension {X.shape[1]} doesn't match expected {self.input_size}"
            )
        
        # Forward propagation step 1: Input → Hidden Layer
        # Linear transformation followed by sigmoid activation
        self.z1 = np.dot(X, self.W1) + self.b1  # Shape: (batch_size, hidden_size)
        self.a1 = self._sigmoid(self.z1)        # Apply non-linearity

        # Forward propagation step 2: Hidden → Output Layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Shape: (batch_size, output_size)
        
        # Choose output activation based on problem type
        if self.output_size > 1:
            # Multi-class classification: use softmax for probability distribution
            self.a2 = self._softmax(self.z2)
        else:
            # Binary classification: use sigmoid for probability
            self.a2 = self._sigmoid(self.z2)
            
        return self.a1, self.a2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for the given input data.
        
        Educational Context:
            Prediction converts the network's continuous outputs to discrete class labels:
            - Binary classification: threshold at 0.5 (sigmoid > 0.5 → class 1)
            - Multi-class: argmax (highest probability wins)
            This demonstrates how neural networks make discrete decisions
            from continuous probability distributions.
        
        Args:
            X: Input data of shape (n_samples, input_size)
            
        Returns:
            Predicted class labels of shape (n_samples,)
            - Binary: 0 or 1
            - Multi-class: 0, 1, ..., num_classes-1
            
        Raises:
            ValueError: If input dimensions are incorrect
        """
        # Validate input format
        if X.ndim != 2:
            raise ValueError(f"Input must be 2D array, got {X.ndim}D")
        
        # Forward pass to get predictions
        _, output = self._forward(X)
        
        if self.output_size > 1:
            # Multi-class: return the index of the highest probability
            # argmax along axis=1 gives the predicted class for each sample
            return np.argmax(output, axis=1)
        else:
            # Binary classification: threshold sigmoid output at 0.5
            return (output > 0.5).astype(int).flatten()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score on given data.
        
        Educational Context:
            Accuracy is the simplest classification metric: fraction of correct predictions.
            While not always the best metric (especially for imbalanced datasets),
            it provides an intuitive measure of model performance.
            
        Args:
            X: Input features of shape (n_samples, input_size)
            y: True labels (can be one-hot encoded or class indices)
            
        Returns:
            Accuracy score between 0.0 and 1.0 (higher is better)
        """
        # Get model predictions
        predictions = self.predict(X)
        
        # Handle one-hot encoded labels by converting to class indices
        if y.ndim > 1 and y.shape[1] > 1:
            y_labels = np.argmax(y, axis=1)
        else:
            y_labels = y.flatten()
            
        # Calculate accuracy as fraction of correct predictions
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
        """Trains the MLP model using stochastic gradient descent with backpropagation.
        
        Educational Context:
            This method implements the backpropagation algorithm, the cornerstone of deep learning:
            
            1. **Forward Pass**: Compute predictions by propagating inputs through the network
            2. **Loss Calculation**: Measure how far predictions are from true labels
            3. **Backward Pass**: Use the chain rule to compute gradients of loss w.r.t. weights
            4. **Weight Update**: Adjust weights in the direction that reduces loss
            
            The Algorithm:
            for each epoch:
                for each sample:
                    1. Forward: a2 = softmax(W2 @ sigmoid(W1 @ x + b1) + b2)
                    2. Loss: L = -log(a2[true_class]) for cross-entropy
                    3. Backward: δ2 = a2 - y, δ1 = (δ2 @ W2.T) ⊙ σ'(z1)
                    4. Update: W2 -= lr * a1.T @ δ2, W1 -= lr * x.T @ δ1
            
            Why SGD (Sample-by-Sample)?
            - Simpler to understand and implement
            - More frequent weight updates
            - Historically important algorithm
            - Demonstrates gradient computation clearly
            
        Args:
            X: Training input samples of shape (n_samples, input_size)
            y: Target values of shape (n_samples, output_size) - one-hot for multi-class
            
        Raises:
            ValueError: If input dimensions are incorrect
        """
        # Validate input dimensions for proper training
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if X.shape[1] != self.input_size:
            raise ValueError(
                f"X feature dimension {X.shape[1]} doesn't match model input_size {self.input_size}"
            )
        
        logger.info(f"Starting SGD training: {self.epochs} epochs on {X.shape[0]} samples")
        logger.info(f"Architecture: {self.input_size}→{self.hidden_size}→{self.output_size}")
        
        # Initialize training tracking
        self.losses = []
        n_samples = X.shape[0]
        global_step = 0

        # Main training loop: iterate over epochs
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            # Stochastic Gradient Descent: process one sample at a time
            for sample_idx in range(n_samples):
                global_step += 1

                # Progress logging for long training runs
                if (sample_idx + 1) % 10000 == 0:
                    logger.info(f"  Epoch {epoch+1}/{self.epochs}, Sample {sample_idx+1}/{n_samples}")

                # Extract single sample (keep as 2D for matrix operations)
                x_sample = X[sample_idx:sample_idx+1]  # Shape: (1, input_size)
                y_sample = y[sample_idx:sample_idx+1]  # Shape: (1, output_size)

                # === FORWARD PASS ===
                # Propagate input through the network to get predictions
                a1, a2 = self._forward(x_sample)

                # === LOSS CALCULATION ===
                # Compute loss for this single sample
                if self.output_size > 1:
                    # Multi-class: Cross-entropy loss
                    # L = -log(p_true_class) where p is predicted probability
                    true_class_idx = np.argmax(y_sample, axis=1)
                    predicted_prob = a2[0, true_class_idx[0]]
                    sample_loss = -np.log(predicted_prob + EPSILON)  # Add epsilon to prevent log(0)
                else:
                    # Binary: Mean Squared Error loss
                    # L = 0.5 * (y - ŷ)²
                    sample_loss = 0.5 * np.mean((y_sample - a2) ** 2)

                epoch_loss += sample_loss

                # Log sample-level metrics for detailed monitoring
                if (self.wandb_run and not self.wandb_run.run.disabled and 
                    (sample_idx + 1) % 100 == 0):
                    self.wandb_run.log(
                        {"Training/Sample_Loss": float(sample_loss)}, step=global_step
                    )

                # === BACKWARD PASS (BACKPROPAGATION) ===
                # Compute gradients using the chain rule
                
                # Step 1: Compute output layer error (δ2)
                if self.output_size > 1:
                    # For softmax + cross-entropy, the gradient simplifies beautifully:
                    # ∂L/∂z2 = (predicted_probabilities - true_labels)
                    delta_output = a2 - y_sample
                else:
                    # For sigmoid + MSE: ∂L/∂z2 = (ŷ - y) * σ'(z2)
                    delta_output = (a2 - y_sample) * self._sigmoid_derivative(a2)

                # Step 2: Propagate error to hidden layer (δ1)
                # Chain rule: ∂L/∂z1 = (∂L/∂z2) * (∂z2/∂a1) * (∂a1/∂z1)
                #                     = δ2 * W2.T * σ'(z1)
                error_hidden = delta_output.dot(self.W2.T)
                delta_hidden = error_hidden * self._sigmoid_derivative(a1)

                # Step 3: Compute weight gradients
                # For weights: ∂L/∂W = (input_to_layer).T @ (error_of_layer)
                # For biases: ∂L/∂b = sum(error_of_layer)
                grad_W2 = a1.T.dot(delta_output)  # Hidden activations × output errors
                grad_b2 = np.sum(delta_output, axis=0, keepdims=True)
                grad_W1 = x_sample.T.dot(delta_hidden)  # Input × hidden errors
                grad_b1 = np.sum(delta_hidden, axis=0, keepdims=True)

                # === WEIGHT UPDATES ===
                # Gradient descent: parameters -= learning_rate * gradients
                self.W1 -= self.learning_rate * grad_W1
                self.b1 -= self.learning_rate * grad_b1
                self.W2 -= self.learning_rate * grad_W2
                self.b2 -= self.learning_rate * grad_b2

            # === EPOCH COMPLETION ===
            # Calculate average loss for this epoch
            epoch_loss = epoch_loss / n_samples
            self.losses.append(float(epoch_loss))
            
            # Log epoch-level metrics
            if (self.wandb_run and not self.wandb_run.run.disabled):
                self.wandb_run.log({"Training/Epoch_Loss": float(epoch_loss)}, step=global_step)
                
            # Progress reporting
            if (epoch + 1) % max(1, self.epochs // 10) == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.6f}")
        
        logger.info(f"Training completed! Final loss: {self.losses[-1]:.6f}")
        logger.info(f"Total gradient updates: {global_step:,}")