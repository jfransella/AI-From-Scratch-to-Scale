# -*- coding: utf-8 -*-
"""Evaluation utilities for the Perceptron model.

This module provides functions to evaluate trained Perceptron models,
including accuracy calculation, confusion matrix generation, and other
classification metrics.
"""

import numpy as np
from typing import Tuple, Dict, Any, Protocol
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelProtocol(Protocol):
    """Protocol for model interface."""
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict method signature."""
        ...


def evaluate_model(model: ModelProtocol, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Evaluate a trained Perceptron model on given data.
    
    Args:
        model: The trained Perceptron model with a predict method.
        X: Input features of shape (n_samples, n_features).
        y: True labels of shape (n_samples,).
        
    Returns:
        Dictionary containing evaluation metrics.
        
    Raises:
        ValueError: If inputs are invalid or incompatible.
    """
    # Input validation
    if not hasattr(model, 'predict'):
        raise ValueError("Model must have a 'predict' method")
    if not isinstance(X, np.ndarray):
        raise ValueError(f"X must be a numpy array, got {type(X)}")
    if not isinstance(y, np.ndarray):
        raise ValueError(f"y must be a numpy array, got {type(y)}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got {X.ndim}D")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D array, got {y.ndim}D")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
    if X.shape[0] == 0:
        raise ValueError("Cannot evaluate on empty dataset")
    
    predictions = model.predict(X)
    
    # Validate predictions
    if predictions.shape != y.shape:
        raise ValueError(f"Prediction shape {predictions.shape} doesn't match target shape {y.shape}")
    
    metrics = {
        'accuracy': accuracy_score(y, predictions),
        'precision': precision_score(y, predictions, zero_division=0),
        'recall': recall_score(y, predictions, zero_division=0),
        'f1_score': f1_score(y, predictions, zero_division=0)
    }
    
    return metrics


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.
    
    Args:
        y_true: True labels of shape (n_samples,).
        y_pred: Predicted labels of shape (n_samples,).
        
    Returns:
        Accuracy score as a float between 0 and 1.
        
    Raises:
        ValueError: If input arrays are invalid or incompatible.
    """
    if not isinstance(y_true, np.ndarray):
        raise ValueError(f"y_true must be a numpy array, got {type(y_true)}")
    if not isinstance(y_pred, np.ndarray):
        raise ValueError(f"y_pred must be a numpy array, got {type(y_pred)}")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot compute accuracy on empty arrays")
    
    return np.mean(y_true == y_pred)