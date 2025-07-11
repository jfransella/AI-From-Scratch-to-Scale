"""Weights & Biases integration for Perceptron model.

This module provides Perceptron-specific W&B logging capabilities by extending
the shared base class. It focuses on binary classification visualizations and
learning dynamics specific to the perceptron algorithm.

Educational Objectives:
- Understand professional ML experiment tracking patterns
- Learn separation of concerns in software architecture
- Practice inheritance with abstract base classes
- Visualize binary classification decision boundaries
- Track convergence behavior of iterative learning algorithms
"""

from typing import Dict, Any, Optional, List
import numpy as np
import logging
import sys
import os
import matplotlib.pyplot as plt

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import from standardized shared package
from ai_from_scratch_shared import BaseWandbVisualizer, initialize_wandb, finish_wandb

# Import the new shared framework visualizer
from visualize import PerceptronVisualizer

logger = logging.getLogger(__name__)


class PerceptronWandbVisualizer(BaseWandbVisualizer):
    """Perceptron-specific W&B visualization and experiment tracking.
    
    This class extends BaseWandbVisualizer to provide specialized logging
    and visualization capabilities for Perceptron experiments, focusing on:
    - Binary classification decision boundaries
    - Learning curve visualization
    - Weight evolution tracking
    - Educational insights about linear separability
    """
    
    def __init__(self, wandb_run: Optional[Any] = None, enabled: bool = True) -> None:
        """Initialize the Perceptron W&B visualizer.
        
        Args:
            wandb_run: Active Weights & Biases run object
            enabled: Whether to enable W&B logging
        """
        super().__init__(wandb_run, enabled)
        
        # Initialize the shared framework visualizer
        self.visualizer = PerceptronVisualizer(enabled=enabled)
        
        logger.info(f"Perceptron W&B visualizer initialized - {'enabled' if enabled else 'disabled'}")
    
    def log_model_config(self, config: Dict[str, Any]) -> None:
        """Log perceptron model configuration (implements abstract method).
        
        Args:
            config: Dictionary containing model configuration including
                   learning_rate, max_epochs, tolerance, etc.
        """
        # Extract perceptron-specific configuration
        perceptron_config = {
            "model_type": "Perceptron",
            "algorithm": "Binary Classification",
            "learning_rule": "Perceptron Learning Rule",
            "activation": "Step Function",
            **config  # Include all provided configuration
        }
        
        if self.enabled:
            self.wandb_run.config.update(perceptron_config)
            logger.info(f"Logged perceptron configuration: {list(perceptron_config.keys())}")
    
    def log_training_progress(self, metrics: Dict[str, Any], step: int) -> None:
        """Log training progress metrics (implements abstract method).
        
        Args:
            metrics: Training metrics including loss, accuracy, weight updates
            step: Current training epoch/iteration
        """
        if self.enabled:
            self.wandb_run.log(metrics, step=step)
            logger.debug(f"Logged training progress at step {step}: {list(metrics.keys())}")
    
    def create_model_visualizations(self, **kwargs) -> None:
        """Create perceptron-specific visualizations (implements abstract method).
        
        Args:
            **kwargs: Visualization parameters including model, data, predictions
        """
        model = kwargs.get('model')
        X = kwargs.get('X')
        y = kwargs.get('y')
        predictions = kwargs.get('predictions')
        
        if model is not None and X is not None and y is not None:
            # Create decision boundary visualization
            self._log_decision_boundary(model, X, y)
        
        if model is not None and hasattr(model, 'losses_'):
            # Create learning curve
            self._log_learning_curve(model.losses_)
        
        if model is not None:
            # Log weight evolution
            self._log_weight_analysis(model)
        
        if y is not None and predictions is not None:
            # Log classification metrics
            self._log_classification_metrics(y, predictions)
    
    def log_training_results(self, model, X: np.ndarray, y: np.ndarray, 
                           predictions: np.ndarray, class_names: Optional[List[str]] = None) -> None:
        """Comprehensive logging of training results and visualizations.
        
        Args:
            model: Trained perceptron model
            X: Input features
            y: True labels
            predictions: Model predictions
            class_names: Optional class names for labeling
        """
        try:
            if not self.enabled:
                logger.info("Visualization disabled - skipping training results logging")
                return
            
            logger.info("Logging comprehensive training results...")
            
            # Log model configuration
            model_config = {
                "learning_rate": model.learning_rate,
                "n_iterations": model.n_iters,
                "input_features": X.shape[1],
                "n_samples": X.shape[0],
                "converged": getattr(model, 'converged_', False)
            }
            self.log_model_config(model_config)
            
            # Log classification metrics
            self._log_classification_metrics(y, predictions)
            
            # Generate visualizations
            self._log_decision_boundary(model, X, y)
            if hasattr(model, 'losses_') and model.losses_:
                self._log_learning_curve(model.losses_)
            self._log_weight_analysis(model)
            
            logger.info("Training results logging complete")
            
        except Exception as e:
            logger.error(f"Failed to log training results: {e}")

    def _log_decision_boundary(self, model: Any, X: np.ndarray, y: np.ndarray) -> None:
        """Create and log decision boundary visualization using shared framework."""
        try:
            if not self.enabled or X.shape[1] != 2:
                return  # Only create 2D decision boundaries
            
            # Use the shared framework visualizer
            fig = self.visualizer.plot_decision_boundary(
                model, X, y, 
                title="Perceptron Decision Boundary",
                save_name=None  # Don't save to file, just return figure
            )
            
            if fig is not None:
                self.log_figure(fig, "decision_boundary")
                plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Could not create decision boundary plot: {e}")
    
    def _log_learning_curve(self, losses: list) -> None:
        """Create and log learning curve visualization using shared framework."""
        try:
            if not self.enabled or not losses:
                return
            
            # Use the shared framework visualizer
            fig = self.visualizer.plot_learning_curve(
                losses,
                title="Perceptron Learning Curve", 
                save_name=None  # Don't save to file, just return figure
            )
            
            if fig is not None:
                self.log_figure(fig, "learning_curve")
                plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Could not create learning curve: {e}")
    
    def _log_weight_analysis(self, model: Any) -> None:
        """Log weight analysis and evolution."""
        try:
            if not self.enabled or not hasattr(model, 'weights_'):
                return
            
            # Log final weights
            weight_metrics = {
                "final_weight_norm": float(np.linalg.norm(model.weights_)),
                "final_bias": float(model.bias_) if hasattr(model, 'bias_') else 0.0,
                "weight_magnitude_avg": float(np.mean(np.abs(model.weights_))),
                "weight_magnitude_max": float(np.max(np.abs(model.weights_)))
            }
            
            self.log_metrics(weight_metrics)
            
            # Create weight histogram if we have weight history
            if hasattr(model, 'weight_history_') and model.weight_history_:
                import matplotlib.pyplot as plt
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Weight evolution over time
                weight_history = np.array(model.weight_history_)
                epochs = range(len(weight_history))
                
                for i in range(weight_history.shape[1]):
                    ax1.plot(epochs, weight_history[:, i], label=f'Weight {i+1}', marker='o', markersize=3)
                
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Weight Value')
                ax1.set_title('Weight Evolution During Training')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Final weight distribution
                ax2.hist(model.weights_, bins=10, alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Weight Value')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Final Weight Distribution')
                ax2.grid(True, alpha=0.3)
                
                self.log_figure(fig, "weight_analysis")
                plt.close(fig)
                
        except Exception as e:
            logger.warning(f"Could not create weight analysis: {e}")
    
    def _log_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Log binary classification metrics."""
        try:
            if not self.enabled:
                return
            
            # Calculate basic metrics
            accuracy = np.mean(y_true == y_pred)
            
            # For binary classification, calculate additional metrics
            if len(np.unique(y_true)) == 2:
                tp = np.sum((y_true == 1) & (y_pred == 1))
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                metrics = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "true_positives": int(tp),
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn)
                }
            else:
                metrics = {"accuracy": float(accuracy)}
            
            self.log_metrics(metrics)
            logger.info(f"Logged classification metrics: accuracy={accuracy:.3f}")
            
        except Exception as e:
            logger.warning(f"Could not log classification metrics: {e}")


# Export for backward compatibility
__all__ = ['PerceptronWandbVisualizer', 'initialize_wandb', 'finish_wandb']
