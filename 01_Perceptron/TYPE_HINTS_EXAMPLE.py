# Type hints and validation improvements for model.py

from typing import Optional, List, Tuple
import numpy as np
import logging

# Constants
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_ITERATIONS = 1000
MNIST_IMAGE_SIZE = 784
MNIST_WIDTH = 28
MNIST_HEIGHT = 28
PIXEL_NORMALIZATION_FACTOR = 255.0

class Perceptron:
    """A single-layer Perceptron for binary classification."""
    
    def __init__(
        self, 
        learning_rate: float = DEFAULT_LEARNING_RATE, 
        n_iters: int = DEFAULT_ITERATIONS, 
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Initialize the Perceptron model.
        
        Args:
            learning_rate: Step size for weight updates
            n_iters: Number of training iterations
            logger: Optional logger instance
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if n_iters <= 0:
            raise ValueError("Number of iterations must be positive")
            
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.errors_per_epoch: List[int] = []
        self.logger = logger or logging.getLogger(__name__)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Perceptron model.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            
        Raises:
            ValueError: If input shapes are invalid
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got {y.ndim}D")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
            
        # Implementation continues...
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels of shape (n_samples,)
            
        Raises:
            ValueError: If model is not fitted or input shape is invalid
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Input features ({X.shape[1]}) don't match model features ({self.weights.shape[0]})")
            
        # Implementation continues...
        
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"Perceptron(learning_rate={self.learning_rate}, n_iters={self.n_iters})"
