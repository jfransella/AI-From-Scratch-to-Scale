"""
Evaluation utilities for ADALINE (Adaptive Linear Neuron).

This module provides comprehensive evaluation functions for ADALINE models
including various regression metrics and validation procedures.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import config, ERROR_MESSAGES


logger = logging.getLogger(__name__)


class ADALINEEvaluator:
    """
    Comprehensive evaluator for ADALINE models.
    
    This class provides various evaluation metrics and validation procedures
    for ADALINE regression models.
    """
    
    def __init__(self, cfg: Any = None) -> None:
        """
        Initialize ADALINE evaluator.
        
        Args:
            cfg: Configuration object
        """
        self.config = cfg if cfg is not None else config
        logger.info("ADALINE Evaluator initialized")
    
    def evaluate_predictions(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           metrics: Optional[Tuple[str, ...]] = None) -> Dict[str, float]:
        """
        Evaluate predictions using multiple regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = metrics or self.config.EVALUATION_METRICS
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
        
        results = {}
        
        for metric in metrics:
            try:
                if metric == 'mse':
                    results[metric] = mean_squared_error(y_true, y_pred)
                elif metric == 'mae':
                    results[metric] = mean_absolute_error(y_true, y_pred)
                elif metric == 'r2_score':
                    results[metric] = r2_score(y_true, y_pred)
                elif metric == 'rmse':
                    results[metric] = np.sqrt(mean_squared_error(y_true, y_pred))
                elif metric == 'mape':
                    results[metric] = self._mean_absolute_percentage_error(y_true, y_pred)
                else:
                    logger.warning(f"Unknown metric: {metric}")
                    continue
            except Exception as e:
                logger.error(f"Error computing metric {metric}: {e}")
                results[metric] = np.nan
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def _mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Absolute Percentage Error (MAPE).
        
        MAPE = (1/n) * Σ|(y_true - y_pred) / y_true| * 100
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            MAPE value
        """
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.nan
        
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape
    
    def cross_validate(self, 
                      model: Any,
                      X: np.ndarray,
                      y: np.ndarray,
                      cv_folds: int = 5,
                      random_state: Optional[int] = None) -> Dict[str, float]:
        """
        Perform cross-validation on ADALINE model.
        
        Args:
            model: ADALINE model instance
            X: Input features
            y: Target values
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary of mean and std of cross-validation scores
        """
        from sklearn.model_selection import KFold
        
        random_state = random_state or self.config.RANDOM_SEED
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"Cross-validation fold {fold + 1}/{cv_folds}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model for this fold
            fold_model = type(model)(
                input_size=model.input_size,
                learning_rate=model.learning_rate,
                random_seed=random_state + fold,
                cfg=model.config
            )
            
            # Train model
            fold_model.fit(X_train, y_train, X_val, y_val)
            
            # Evaluate
            y_pred = fold_model.predict(X_val)
            fold_score = r2_score(y_val, y_pred)
            scores.append(fold_score)
            
            logger.info(f"Fold {fold + 1} R² score: {fold_score:.4f}")
        
        # Compute statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        results = {
            'mean_r2_score': mean_score,
            'std_r2_score': std_score,
            'fold_scores': scores
        }
        
        logger.info(f"Cross-validation results: mean={mean_score:.4f}, std={std_score:.4f}")
        return results
    
    def compare_models(self, 
                      models: Dict[str, Any],
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple ADALINE models on test data.
        
        Args:
            models: Dictionary of model name to model instance
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of model names to evaluation metrics
        """
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            
            try:
                y_pred = model.predict(X_test)
                metrics = self.evaluate_predictions(y_test, y_pred)
                results[model_name] = metrics
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def analyze_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Analyze prediction residuals for model diagnostics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of residual statistics
        """
        residuals = y_true - y_pred
        
        analysis = {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'min_residual': np.min(residuals),
            'max_residual': np.max(residuals),
            'median_residual': np.median(residuals),
            'skewness': self._compute_skewness(residuals),
            'kurtosis': self._compute_kurtosis(residuals)
        }
        
        logger.info(f"Residual analysis: {analysis}")
        return analysis
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3


def create_evaluator(config: Any = None) -> ADALINEEvaluator:
    """
    Factory function to create an ADALINE evaluator.
    
    Args:
        config: Configuration object
        
    Returns:
        ADALINEEvaluator instance
    """
    return ADALINEEvaluator(config) 