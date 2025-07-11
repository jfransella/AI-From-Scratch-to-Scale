"""
Perceptron Weights & Biases Integration
======================================

This module provides W&B experiment tracking specifically designed for
Perceptron models, extending the shared base W&B integration framework.

Educational Objectives:
- Demonstrate basic neural network experiment tracking
- Show proper separation of visualization and experiment tracking concerns
- Provide systematic comparison of different datasets and hyperparameters
- Enable reproducible perceptron experiments

Key Features:
- Perceptron-specific visualizations (decision boundaries, learning curves)
- Automatic logging of training dynamics and parameter evolution
- Classification performance tracking and analysis
- Integration with existing visualization functions
"""

import logging
import os
import sys
from typing import Dict, Any, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

# Import the base W&B integration framework
try:
    # Add the project root to Python path for shared imports
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from shared.utils.wandb_integration import BaseWandbVisualizer
    logger = logging.getLogger(__name__)
    logger.info("Using shared W&B integration framework")
except ImportError:
    # Fall back for when shared module is not available
    logger = logging.getLogger(__name__)
    logger.warning("Shared W&B integration not found, using local implementation")
    BaseWandbVisualizer = object  # Fallback to regular class

# Import local visualization functions
try:
    from .visualize import _plot_confusion_matrix, _plot_learning_curve, _plot_decision_boundary
except ImportError:
    from visualize import _plot_confusion_matrix, _plot_learning_curve, _plot_decision_boundary

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class PerceptronWandbVisualizer(BaseWandbVisualizer):
    """
    Perceptron-specific W&B integration extending the base framework.
    
    This class provides specialized experiment tracking for Perceptron models,
    including learning curve analysis, decision boundary visualization, and
    parameter evolution tracking.
    
    Educational Focus:
    - Demonstrates inheritance from a professional base class
    - Shows perceptron-specific metrics and visualizations
    - Provides clean separation between plotting and logging
    - Enables systematic hyperparameter exploration
    """
    
    def __init__(self, wandb_run: Optional[Any] = None, enabled: bool = True) -> None:
        """
        Initialize the Perceptron W&B visualizer.
        
        Args:
            wandb_run: Active W&B run object
            enabled: Whether to enable W&B logging
        """
        super().__init__(wandb_run, enabled, plots_dir="outputs/plots")
        logger.info("Perceptron W&B visualizer initialized")
    
    # =================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS (REQUIRED BY BASE CLASS)
    # =================================================================
    
    def log_model_config(self, config: Dict[str, Any]) -> None:
        """
        Log Perceptron-specific configuration and hyperparameters.
        
        Args:
            config: Model configuration dictionary containing:
                   - learning_rate: Learning rate for weight updates
                   - n_iters: Maximum number of training iterations
                   - experiment_type: Type of experiment (e.g., 'and', 'mnist')
                   
        Educational Focus:
        Shows how to track the minimal set of hyperparameters that
        characterize a perceptron model.
        """
        perceptron_metrics = {
            "model/type": "Perceptron",
            "model/learning_rate": config.get("learning_rate", 0.01),
            "model/max_iterations": config.get("n_iters", 1000),
            "model/experiment_type": config.get("experiment_type", "unknown"),
            "model/total_parameters": config.get("total_parameters", 0)
        }
        
        # Add dataset information if available
        if "dataset_size" in config:
            perceptron_metrics["data/dataset_size"] = config["dataset_size"]
        if "num_features" in config:
            perceptron_metrics["data/num_features"] = config["num_features"]
        if "num_classes" in config:
            perceptron_metrics["data/num_classes"] = config["num_classes"]
        
        self.log_metrics(perceptron_metrics)
        logger.info("Perceptron model configuration logged")
    
    def log_training_progress(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log Perceptron training progress metrics.
        
        Args:
            metrics: Training metrics dictionary containing:
                    - misclassifications: Number of misclassified samples
                    - weights: Current weight values  
                    - bias: Current bias value
                    - accuracy: Current training accuracy (optional)
            step: Training iteration/epoch number
            
        Educational Focus:
        Demonstrates how to track the unique aspects of perceptron learning:
        - Error-driven updates (misclassifications)
        - Parameter evolution over time
        - Convergence detection
        """
        training_metrics = {
            "training/misclassifications": metrics.get("misclassifications", 0),
            "training/step": step
        }
        
        # Add accuracy if available
        if "accuracy" in metrics:
            training_metrics["training/accuracy"] = metrics["accuracy"]
        
        # Log parameter statistics
        if "weights" in metrics:
            weights = np.array(metrics["weights"])
            training_metrics.update({
                "parameters/weights_mean": np.mean(weights),
                "parameters/weights_std": np.std(weights),
                "parameters/weights_norm": np.linalg.norm(weights)
            })
        
        if "bias" in metrics:
            training_metrics["parameters/bias"] = metrics["bias"]
        
        self.log_metrics(training_metrics, step=step)
        
        # Log parameter distributions periodically
        if self.enabled and step % 10 == 0:  # Every 10 steps
            self._log_parameter_distributions(metrics, step)
    
    def create_model_visualizations(self, model=None, X=None, y=None, 
                                  predictions=None, class_names=None, **kwargs) -> None:
        """
        Create Perceptron-specific visualizations and analysis plots.
        
        Args:
            model: Trained Perceptron model
            X: Input features
            y: True labels
            predictions: Model predictions
            class_names: Names of classes for labeling
            **kwargs: Additional visualization parameters
            
        Educational Focus:
        Shows the three key visualizations for understanding perceptron behavior:
        1. Confusion Matrix - Classification performance
        2. Learning Curve - Training dynamics
        3. Decision Boundary - Model's learned decision rule (for 2D data)
        """
        if not (model and X is not None and y is not None and predictions is not None):
            logger.warning("Insufficient data for visualizations")
            return
        
        try:
            # 1. Confusion Matrix
            self.log_confusion_matrix(y, predictions, class_names)
            
            # 2. Learning Curve
            self.log_learning_curve(model.errors_per_epoch if hasattr(model, 'errors_per_epoch') else [])
            
            # 3. Decision Boundary (only for 2D data)
            if X.shape[1] == 2:
                self.log_decision_boundary(model, X, y, class_names)
            
            logger.info("Perceptron visualizations created successfully")
            
        except Exception as e:
            logger.warning(f"Error creating perceptron visualizations: {e}")
    
    # =================================================================
    # PERCEPTRON-SPECIFIC VISUALIZATION METHODS
    # =================================================================
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: Optional[List[str]] = None) -> None:
        """
        Log confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names for labeling
            
        Educational Focus:
        Shows how to analyze classification performance beyond simple accuracy.
        """
        try:
            fig = _plot_confusion_matrix(y_true, y_pred, class_names)
            self.log_figure(fig, "confusion_matrix", close_figure=True)
            
            # Log summary metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            
            confusion_metrics = {
                "evaluation/accuracy": accuracy,
                "evaluation/precision": precision,
                "evaluation/recall": recall,
                "evaluation/f1_score": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            }
            
            self.log_metrics(confusion_metrics)
            
        except Exception as e:
            logger.warning(f"Error logging confusion matrix: {e}")
    
    def log_learning_curve(self, errors_per_epoch: List[int]) -> None:
        """
        Log learning curve showing error reduction over time.
        
        Args:
            errors_per_epoch: List of misclassification counts per epoch
            
        Educational Focus:
        Visualizes the perceptron learning theorem in action - showing
        how errors decrease over time for linearly separable data.
        """
        try:
            if not errors_per_epoch:
                logger.warning("No learning curve data available")
                return
            
            fig = _plot_learning_curve(errors_per_epoch)
            self.log_figure(fig, "learning_curve", close_figure=True)
            
            # Log learning curve statistics
            learning_metrics = {
                "learning/initial_errors": errors_per_epoch[0] if errors_per_epoch else 0,
                "learning/final_errors": errors_per_epoch[-1] if errors_per_epoch else 0,
                "learning/total_epochs": len(errors_per_epoch),
                "learning/converged": errors_per_epoch[-1] == 0 if errors_per_epoch else False,
                "learning/error_reduction": (errors_per_epoch[0] - errors_per_epoch[-1]) if len(errors_per_epoch) > 1 else 0
            }
            
            self.log_metrics(learning_metrics)
            
        except Exception as e:
            logger.warning(f"Error logging learning curve: {e}")
    
    def log_decision_boundary(self, model, X: np.ndarray, y: np.ndarray, 
                            class_names: Optional[List[str]] = None) -> None:
        """
        Log decision boundary visualization for 2D data.
        
        Args:
            model: Trained perceptron model
            X: 2D input features
            y: True labels
            class_names: Optional class names for labeling
            
        Educational Focus:
        Visualizes the linear decision boundary learned by the perceptron,
        helping students understand the geometric interpretation of the model.
        """
        try:
            if X.shape[1] != 2:
                logger.info("Decision boundary visualization only available for 2D data")
                return
            
            fig = _plot_decision_boundary(X, y, model, class_names)
            self.log_figure(fig, "decision_boundary", close_figure=True)
            
            # Log decision boundary characteristics
            if hasattr(model, 'weights') and hasattr(model, 'bias'):
                weights = np.array(model.weights)
                bias = model.bias
                
                boundary_metrics = {
                    "boundary/weight_magnitude": np.linalg.norm(weights),
                    "boundary/bias_value": bias,
                    "boundary/slope": -weights[0] / weights[1] if weights[1] != 0 else float('inf'),
                    "boundary/intercept": -bias / weights[1] if weights[1] != 0 else float('inf')
                }
                
                self.log_metrics(boundary_metrics)
            
        except Exception as e:
            logger.warning(f"Error logging decision boundary: {e}")
    
    def _log_parameter_distributions(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log parameter distribution histograms to W&B.
        
        Args:
            metrics: Metrics containing weights and bias
            step: Current training step
        """
        if not self.enabled or not WANDB_AVAILABLE:
            return
        
        try:
            log_data = {}
            
            # Log weight distribution
            if "weights" in metrics:
                weights = np.array(metrics["weights"])
                log_data["parameters/weights_histogram"] = wandb.Histogram(weights)
            
            # Log bias as a single value
            if "bias" in metrics:
                log_data["parameters/bias_value"] = metrics["bias"]
            
            if log_data:
                self.wandb_run.log(log_data, step=step)
                
        except Exception as e:
            logger.warning(f"Error logging parameter distributions: {e}")
    
    def _extract_summary_metrics(self, results: Dict[str, Any]) -> List[str]:
        """
        Extract key metrics for experiment summary table.
        
        Args:
            results: Experiment results dictionary
            
        Returns:
            List of formatted metric values
        """
        metrics = []
        
        # Final accuracy
        final_accuracy = results.get('final_accuracy', results.get('accuracy', 0))
        metrics.append(f"{final_accuracy:.3f}")
        
        # Training epochs/convergence
        total_epochs = results.get('total_epochs', results.get('epochs', 0))
        converged = results.get('converged', False)
        metrics.append(f"{total_epochs} {'(conv)' if converged else '(max)'}")
        
        # Error reduction
        error_reduction = results.get('error_reduction', 0)
        metrics.append(f"{error_reduction}")
        
        return metrics
    
    def _get_summary_columns(self) -> List[str]:
        """
        Get column names for summary table.
        
        Returns:
            List of column names
        """
        return ["Final Accuracy", "Training Epochs", "Error Reduction"]


# =================================================================
# UTILITY FUNCTIONS FOR BACKWARD COMPATIBILITY
# =================================================================

def initialize_perceptron_wandb(project_name: str, config: Dict[str, Any], 
                              enabled: bool = True) -> tuple:
    """
    Initialize W&B for perceptron experiments.
    
    Args:
        project_name: W&B project name
        config: Experiment configuration
        enabled: Whether to enable W&B logging
        
    Returns:
        Tuple of (wandb_run, perceptron_visualizer)
    """
    try:
        if not enabled or not WANDB_AVAILABLE:
            return None, PerceptronWandbVisualizer(enabled=False)
        
        wandb_run = wandb.init(
            project=project_name,
            config=config,
            mode="online" if enabled else "disabled"
        )
        
        visualizer = PerceptronWandbVisualizer(wandb_run, enabled=True)
        return wandb_run, visualizer
        
    except Exception as e:
        logger.warning(f"Failed to initialize perceptron W&B: {e}")
        return None, PerceptronWandbVisualizer(enabled=False)


def finish_perceptron_wandb(wandb_run) -> None:
    """
    Finish perceptron W&B run gracefully.
    
    Args:
        wandb_run: W&B run object to finish
    """
    if wandb_run is not None:
        try:
            wandb_run.finish()
            logger.info("Perceptron W&B run finished successfully")
        except Exception as e:
            logger.warning(f"Error finishing perceptron W&B run: {e}")
