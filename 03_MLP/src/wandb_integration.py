"""Weights & Biases integration for MLP model.

This module provides MLP-specific W&B logging capabilities by extending
the shared base class. It focuses on multi-layer neural network visualizations
including architecture analysis, weight distributions, and learning dynamics.

Educational Objectives:
- Understand multi-layer network experiment tracking
- Learn advanced neural network visualization techniques
- Practice professional ML experiment organization
- Visualize weight evolution in deep architectures
- Track multi-class classification performance
"""

from typing import Dict, Any, Optional, List
import numpy as np
import logging

# Import from standardized shared package
from ai_from_scratch_shared import BaseWandbVisualizer, initialize_wandb, finish_wandb

logger = logging.getLogger(__name__)


class MLPWandbVisualizer(BaseWandbVisualizer):
    """MLP-specific W&B visualization and experiment tracking.
    
    This class handles all W&B logging for MLP experiments including:
    - Training metrics and loss curves
    - Confusion matrices and classification reports
    - Learning curves and convergence analysis
    - Network architecture visualization
    - Decision boundaries (for 2D data)
    - Weight and activation histograms
    
    Educational Context:
        MLPs introduce several concepts beyond perceptrons:
        - Multi-layer weight visualization
        - Hidden layer activation analysis
        - Convergence behavior tracking
        - Loss landscape visualization
        - Gradient flow monitoring
    """
    
    def __init__(self):
        """Initialize the MLP W&B visualizer."""
        super().__init__()
        self.model_type = "MLP"
    
    def log_training_results(self, model: Any, X: np.ndarray, y: np.ndarray, 
                           predictions: np.ndarray, class_names: Optional[List[str]] = None,
                           **kwargs) -> None:
        """Log comprehensive MLP training results to W&B.
        
        Args:
            model: Trained MLP model instance
            X: Input features used for training
            y: True labels 
            predictions: Model predictions
            class_names: Names of the classes for visualization
            **kwargs: Additional logging parameters
        """
        if not self._check_wandb_available():
            logger.info("W&B not available, skipping logging")
            return
        
        logger.info("Logging MLP training results to W&B...")
        
        # 1. Log confusion matrix
        self._log_confusion_matrix(y, predictions, class_names)
        
        # 2. Log learning curves
        self._log_learning_curves(model)
        
        # 3. Log network architecture info
        self._log_architecture_info(model)
        
        # 4. Log decision boundary (if 2D data)
        if X.shape[1] == 2:
            self._log_decision_boundary(model, X, y, class_names)
        
        # 5. Log weight histograms
        self._log_weight_histograms(model)
        
        # 6. Log final metrics summary
        self._log_final_metrics(y, predictions)
        
        logger.info("MLP results logged successfully to W&B")
    
    def _log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: Optional[List[str]] = None) -> None:
        """Create and log confusion matrix."""
        try:
            from .visualize import plot_confusion_matrix
            
            fig = plot_confusion_matrix(y_true, y_pred, class_names)
            self._log_plot("Plots/Confusion_Matrix", fig)
            
        except ImportError as e:
            logger.warning(f"Could not import visualization functions: {e}")
    
    def _log_learning_curves(self, model: Any) -> None:
        """Log learning curves showing loss over epochs."""
        try:
            from .visualize import plot_learning_curve
            
            if hasattr(model, 'loss_history') and model.loss_history:
                fig = plot_learning_curve(model.loss_history)
                self._log_plot("Plots/Learning_Curve", fig)
            else:
                logger.warning("Model has no loss_history attribute for learning curve")
                
        except ImportError as e:
            logger.warning(f"Could not import learning curve visualization: {e}")
    
    def _log_architecture_info(self, model: Any) -> None:
        """Log network architecture information."""
        try:
            architecture_info = {
                "input_size": getattr(model, 'input_size', 'unknown'),
                "hidden_size": getattr(model, 'hidden_size', 'unknown'), 
                "output_size": getattr(model, 'output_size', 'unknown'),
                "learning_rate": getattr(model, 'learning_rate', 'unknown'),
                "epochs": getattr(model, 'epochs', 'unknown'),
                "total_parameters": self._count_parameters(model)
            }
            
            self._log_dict("Architecture", architecture_info)
            
        except Exception as e:
            logger.warning(f"Could not log architecture info: {e}")
    
    def _log_decision_boundary(self, model: Any, X: np.ndarray, y: np.ndarray,
                             class_names: Optional[List[str]] = None) -> None:
        """Log decision boundary visualization for 2D data."""
        try:
            from .visualize import plot_decision_boundary
            
            fig = plot_decision_boundary(model, X, y, class_names)
            if fig is not None:
                self._log_plot("Plots/Decision_Boundary", fig)
            
        except ImportError as e:
            logger.warning(f"Could not import decision boundary visualization: {e}")
    
    def _log_weight_histograms(self, model: Any) -> None:
        """Log histograms of network weights."""
        try:
            import matplotlib.pyplot as plt
            
            # Log hidden layer weights
            if hasattr(model, 'W1') and model.W1 is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(model.W1.flatten(), bins=50, alpha=0.7, label='Hidden Weights (W1)')
                ax.set_xlabel('Weight Value')
                ax.set_ylabel('Frequency')
                ax.set_title('Hidden Layer Weight Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
                self._log_plot("Plots/Weight_Histogram_Hidden", fig)
            
            # Log output layer weights  
            if hasattr(model, 'W2') and model.W2 is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(model.W2.flatten(), bins=50, alpha=0.7, label='Output Weights (W2)', color='orange')
                ax.set_xlabel('Weight Value')
                ax.set_ylabel('Frequency')
                ax.set_title('Output Layer Weight Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
                self._log_plot("Plots/Weight_Histogram_Output", fig)
                
        except ImportError as e:
            logger.warning(f"Could not create weight histograms: {e}")
    
    def _log_final_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Log final performance metrics."""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Handle both binary and multi-class cases
            average_method = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
            
            metrics = {
                "final_accuracy": float(accuracy_score(y_true, y_pred)),
                "final_precision": float(precision_score(y_true, y_pred, average=average_method, zero_division=0)),
                "final_recall": float(recall_score(y_true, y_pred, average=average_method, zero_division=0)),
                "final_f1": float(f1_score(y_true, y_pred, average=average_method, zero_division=0))
            }
            
            self._log_dict("Final_Metrics", metrics)
            
        except ImportError as e:
            logger.warning(f"Could not compute final metrics: {e}")
    
    def _count_parameters(self, model: Any) -> int:
        """Count total trainable parameters in the model."""
        try:
            param_count = 0
            
            # Count weights and biases
            if hasattr(model, 'W1') and model.W1 is not None:
                param_count += model.W1.size
            if hasattr(model, 'b1') and model.b1 is not None:
                param_count += model.b1.size
            if hasattr(model, 'W2') and model.W2 is not None:
                param_count += model.W2.size
            if hasattr(model, 'b2') and model.b2 is not None:
                param_count += model.b2.size
                
            return param_count
            
        except Exception:
            return 0


# Backward compatibility alias
Visualizer = MLPWandbVisualizer
