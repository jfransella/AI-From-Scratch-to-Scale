"""
ADALINE (Adaptive Linear Neuron) implementation from scratch.

This module implements the ADALINE algorithm with comprehensive mathematical
derivations, proper error handling, and educational documentation. ADALINE
introduces continuous outputs and the Least Mean Squares (LMS) algorithm.
"""

import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

from src.config import config, ERROR_MESSAGES, SUCCESS_MESSAGES, ADALINE_CONSTANTS


logger = logging.getLogger(__name__)


@dataclass
class ADALINEState:
    """State information for ADALINE training and evaluation."""
    weights: np.ndarray
    bias: float
    training_loss: np.ndarray
    validation_loss: np.ndarray
    training_accuracy: np.ndarray
    validation_accuracy: np.ndarray
    weight_history: np.ndarray
    bias_history: np.ndarray
    convergence_epoch: Optional[int] = None
    final_loss: Optional[float] = None


class ADALINE:
    """
    ADALINE (Adaptive Linear Neuron) implementation from scratch.
    
    ADALINE is an improvement over the Perceptron that uses continuous outputs
    and the Least Mean Squares (LMS) algorithm for training. Unlike the Perceptron,
    ADALINE can learn from all training examples in each epoch, not just misclassified ones.
    
    Mathematical Background:
    - Output: y = w^T * x + b (linear combination)
    - Loss: L = (1/2) * Σ(y_true - y_pred)^2 (Mean Squared Error)
    - Update Rule: w_new = w_old + α * (y_true - y_pred) * x
                   b_new = b_old + α * (y_true - y_pred)
    
    where α is the learning rate.
    """
    
    def __init__(self, 
                 input_size: int,
                 learning_rate: float = 0.01,
                 random_seed: Optional[int] = None,
                 cfg: Any = None) -> None:
        """
        Initialize ADALINE model.
        
        Args:
            input_size: Number of input features (including bias term if added)
            learning_rate: Learning rate for gradient descent
            random_seed: Random seed for reproducibility
            cfg: Configuration object (optional)
            
        Raises:
            ValueError: If parameters are invalid
        """
        self.config = cfg if cfg is not None else config
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed or self.config.RANDOM_SEED
        
        # Validate parameters
        self._validate_parameters()
        
        # Initialize model parameters
        self.weights = None
        self.bias = None
        self.is_fitted = False
        
        # Training history
        self.training_loss = []
        self.validation_loss = []
        self.weight_history = []
        self.bias_history = []
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        logger.info(f"ADALINE initialized: input_size={input_size}, "
                   f"learning_rate={learning_rate}, random_seed={self.random_seed}")
    
    def _validate_parameters(self) -> None:
        """Validate model parameters."""
        if self.input_size <= 0:
            raise ValueError(f"Input size must be positive, got {self.input_size}")
        if self.learning_rate <= 0:
            raise ValueError(ERROR_MESSAGES['invalid_learning_rate'].format(self.learning_rate))
    
    def _initialize_parameters(self) -> None:
        """
        Initialize model parameters using Xavier/Glorot initialization.
        
        The weights are initialized with small random values to break symmetry
        and ensure different neurons learn different features.
        """
        # Xavier/Glorot initialization for better gradient flow
        scale = np.sqrt(2.0 / self.input_size)
        
        # Initialize weights with small random values
        self.weights = np.random.normal(
            loc=0.0,
            scale=scale,
            size=(self.input_size,)
        )
        
        # Initialize bias to small random value (if not already included in weights)
        self.bias = np.random.normal(loc=0.0, scale=0.01)
        
        logger.info(f"Parameters initialized: weights={self.weights.shape}, bias={self.bias}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the ADALINE network.
        
        Mathematical formulation:
        y = w^T * x + b
        
        where:
        - w is the weight vector
        - x is the input vector (including bias term if added)
        - b is the bias term (0 if bias term is already in input)
        - y is the continuous output
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Continuous outputs of shape (n_samples,)
            
        Raises:
            ValueError: If input shape is invalid or model not initialized
        """
        if self.weights is None:
            raise ValueError("Model parameters not initialized. Call fit() first.")
        
        if X.shape[1] != self.input_size:
            raise ValueError(ERROR_MESSAGES['dimension_mismatch'].format(
                X.shape[1], self.input_size))
        
        # Linear combination: y = w^T * x + b
        # If bias term is already in input, b = 0
        # X has shape (n_samples, n_features)
        # weights has shape (n_features,)
        # bias is a scalar
        outputs = np.dot(X, self.weights) + self.bias
        
        logger.debug(f"Forward pass: input_shape={X.shape}, output_shape={outputs.shape}")
        return outputs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained ADALINE model.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError(ERROR_MESSAGES['model_not_fitted'])
        
        return self.forward(X)
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Squared Error (MSE) loss.
        
        Mathematical formulation:
        L = (1/2) * Σ(y_true - y_pred)^2
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Mean squared error loss
        """
        # MSE loss: L = (1/2) * Σ(y_true - y_pred)^2
        mse = np.mean(0.5 * (y_true - y_pred) ** 2)
        return mse
    
    def _compute_gradients(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weight and bias updates.
        
        Mathematical derivation:
        ∂L/∂w = ∂/∂w[(1/2) * Σ(y_true - y_pred)^2]
               = Σ(y_true - y_pred) * ∂/∂w(y_pred)
               = Σ(y_true - y_pred) * x
        
        ∂L/∂b = ∂/∂b[(1/2) * Σ(y_true - y_pred)^2]
               = Σ(y_true - y_pred) * ∂/∂b(y_pred)
               = Σ(y_true - y_pred)
        
        Args:
            X: Input features
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Tuple of (weight_gradients, bias_gradient)
        """
        # Compute error: error = y_true - y_pred
        error = y_true - y_pred
        
        # Gradient for weights: ∂L/∂w = Σ(error * x)
        # X has shape (n_samples, n_features)
        # error has shape (n_samples,)
        # We need to compute Σ(error_i * x_i) for each feature
        weight_gradients = np.dot(X.T, error) / X.shape[0]
        
        # Gradient for bias: ∂L/∂b = Σ(error)
        bias_gradient = np.mean(error)
        
        logger.debug(f"Gradients computed: weight_gradients={weight_gradients.shape}, "
                    f"bias_gradient={bias_gradient}")
        
        return weight_gradients, bias_gradient
    
    def _update_parameters(self, weight_gradients: np.ndarray, bias_gradient: float) -> None:
        """
        Update model parameters using gradient descent.
        
        Update rules:
        w_new = w_old - α * ∂L/∂w
        b_new = b_old - α * ∂L/∂b
        
        where α is the learning rate.
        
        Args:
            weight_gradients: Gradients for weights
            bias_gradient: Gradient for bias
        """
        # Update weights: w_new = w_old - α * ∂L/∂w
        self.weights -= self.learning_rate * weight_gradients
        
        # Update bias: b_new = b_old - α * ∂L/∂b
        self.bias -= self.learning_rate * bias_gradient
        
        logger.debug(f"Parameters updated: learning_rate={self.learning_rate}")
    
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            max_epochs: Optional[int] = None,
            convergence_threshold: Optional[float] = None) -> ADALINEState:
        """
        Fit the ADALINE model to the training data.
        
        This method implements the Delta Rule (LMS algorithm) with early stopping
        and comprehensive training history tracking.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            max_epochs: Maximum number of training epochs
            convergence_threshold: Loss threshold for convergence
            
        Returns:
            ADALINEState object containing training history
            
        Raises:
            ValueError: If input data is invalid
        """
        if X.size == 0 or y.size == 0:
            raise ValueError(ERROR_MESSAGES['data_empty'])
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Sample count mismatch: X={X.shape[0]}, y={y.shape[0]}")
        
        if X.shape[1] != self.input_size:
            raise ValueError(ERROR_MESSAGES['dimension_mismatch'].format(
                X.shape[1], self.input_size))
        
        max_epochs = max_epochs or self.config.MAX_EPOCHS
        convergence_threshold = convergence_threshold or self.config.CONVERGENCE_THRESHOLD
        
        logger.info(f"Starting ADALINE training: epochs={max_epochs}, "
                   f"convergence_threshold={convergence_threshold}")
        
        # Initialize parameters if not already done
        if self.weights is None:
            self._initialize_parameters()
        
        # Initialize training history
        self.training_loss = []
        self.validation_loss = []
        self.weight_history = []
        self.bias_history = []
        
        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        max_patience = self.config.MAX_ITERATIONS_WITHOUT_IMPROVEMENT
        
        # Training loop
        for epoch in range(max_epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute training loss
            train_loss = self._compute_loss(y, y_pred)
            self.training_loss.append(train_loss)
            
            # Compute validation loss if validation data provided
            val_loss = None
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self._compute_loss(y_val, y_val_pred)
                self.validation_loss.append(val_loss)
            
            # Store parameter history
            self.weight_history.append(self.weights.copy())
            self.bias_history.append(self.bias)
            
            # Log progress
            if epoch % self.config.LOG_EVERY_N_EPOCHS == 0:
                log_msg = f"Epoch {epoch+1}/{max_epochs}: train_loss={train_loss:.6f}"
                if val_loss is not None:
                    log_msg += f", val_loss={val_loss:.6f}"
                logger.info(log_msg)
            
            # Check convergence
            current_loss = val_loss if val_loss is not None else train_loss
            if current_loss < convergence_threshold:
                logger.info(SUCCESS_MESSAGES['training_converged'].format(epoch + 1))
                break
            
            # Early stopping check
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Compute gradients
            weight_gradients, bias_gradient = self._compute_gradients(X, y, y_pred)
            
            # Update parameters
            self._update_parameters(weight_gradients, bias_gradient)
        
        # Mark model as fitted
        self.is_fitted = True
        
        # Create training state
        training_state = ADALINEState(
            weights=self.weights.copy(),
            bias=self.bias,
            training_loss=np.array(self.training_loss),
            validation_loss=np.array(self.validation_loss) if self.validation_loss else np.array([]),
            training_accuracy=np.array([]),  # Not applicable for regression
            validation_accuracy=np.array([]),  # Not applicable for regression
            weight_history=np.array(self.weight_history),
            bias_history=np.array(self.bias_history),
            convergence_epoch=len(self.training_loss),
            final_loss=self.training_loss[-1] if self.training_loss else None
        )
        
        logger.info(f"Training completed: final_loss={training_state.final_loss:.6f}, "
                   f"epochs={training_state.convergence_epoch}")
        
        return training_state
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the R² score (coefficient of determination).
        
        R² measures the proportion of variance in the dependent variable
        that is predictable from the independent variable(s).
        
        Args:
            X: Input features
            y: True target values
            
        Returns:
            R² score (higher is better, max is 1.0)
        """
        if not self.is_fitted:
            raise ValueError(ERROR_MESSAGES['model_not_fitted'])
        
        y_pred = self.predict(X)
        
        # R² = 1 - (SS_res / SS_tot)
        # where SS_res = Σ(y_true - y_pred)²
        # and SS_tot = Σ(y_true - y_mean)²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add small epsilon to prevent division by zero
        return r2
    
    def save_model(self, filepath: str) -> None:
        """
        Save model parameters to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError(ERROR_MESSAGES['model_not_fitted'])
        
        try:
            np.savez(filepath,
                     weights=self.weights,
                     bias=self.bias,
                     input_size=self.input_size,
                     learning_rate=self.learning_rate,
                     random_seed=self.random_seed)
            logger.info(SUCCESS_MESSAGES['model_saved'].format(filepath))
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """
        Load model parameters from file.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            data = np.load(filepath)
            self.weights = data['weights']
            self.bias = data['bias']
            self.input_size = int(data['input_size'])
            self.learning_rate = float(data['learning_rate'])
            self.random_seed = int(data['random_seed'])
            self.is_fitted = True
            
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get model parameters.
        
        Returns:
            Dictionary containing weights and bias
        """
        if not self.is_fitted:
            raise ValueError(ERROR_MESSAGES['model_not_fitted'])
        
        return {
            'weights': self.weights.copy(),
            'bias': self.bias
        }
    
    def set_parameters(self, weights: np.ndarray, bias: float) -> None:
        """
        Set model parameters.
        
        Args:
            weights: Weight vector
            bias: Bias term
        """
        if weights.shape != (self.input_size,):
            raise ValueError(f"Weight shape {weights.shape} doesn't match input size {self.input_size}")
        
        self.weights = weights.copy()
        self.bias = bias
        self.is_fitted = True
        
        logger.info("Model parameters set")
    
    def __repr__(self) -> str:
        """String representation of the ADALINE model."""
        return (f"ADALINE(input_size={self.input_size}, "
                f"learning_rate={self.learning_rate}, "
                f"fitted={self.is_fitted})")


def create_adaline_model(input_size: int,
                        learning_rate: float = 0.01,
                        random_seed: Optional[int] = None,
                        cfg: Any = None) -> ADALINE:
    """
    Factory function to create an ADALINE model.
    
    Args:
        input_size: Number of input features
        learning_rate: Learning rate for training
        random_seed: Random seed for reproducibility
        cfg: Configuration object
        
    Returns:
        ADALINE model instance
    """
    return ADALINE(
        input_size=input_size,
        learning_rate=learning_rate,
        random_seed=random_seed,
        cfg=cfg
    ) 