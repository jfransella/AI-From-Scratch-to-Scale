"""Weights & Biases integration for ADALINE model.

This module provides ADALINE-specific W&B logging capabilities by extending
the shared base class. It focuses on regression visualizations and learning
dynamics specific to the ADALINE algorithm.

Educational Objectives:
- Understand professional ML experiment tracking patterns
- Learn separation of concerns in software architecture
- Practice inheritance with abstract base classes
- Visualize regression learning dynamics
- Track convergence behavior of continuous learning algorithms
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

# Import the ADALINE visualizer
from visualize import ADALINEVisualizer

logger = logging.getLogger(__name__)


class ADALINEWandbVisualizer(BaseWandbVisualizer):
    """ADALINE-specific W&B visualization and experiment tracking.
    
    This class extends BaseWandbVisualizer to provide specialized logging
    and visualization capabilities for ADALINE experiments, focusing on:
    - Regression learning curves
    - Weight evolution tracking
    - Decision boundary visualization (for 2D data)
    - Educational insights about continuous error-based learning
    """
    
    def __init__(self, wandb_run: Optional[Any] = None, enabled: bool = True) -> None:
        """Initialize the ADALINE W&B visualizer.
        
        Args:
            wandb_run: Active Weights & Biases run object
            enabled: Whether to enable W&B logging
        """
        super().__init__(wandb_run, enabled)
        
        # Initialize the ADALINE visualizer (no enabled argument)
        self.visualizer = ADALINEVisualizer()
        
        logger.info(f"ADALINE W&B visualizer initialized - {'enabled' if enabled else 'disabled'}")
    
    def log_model_config(self, config: Dict[str, Any]) -> None:
        """Log ADALINE model configuration (implements abstract method).
        
        Args:
            config: Dictionary containing model configuration including
                   learning_rate, max_epochs, convergence_threshold, etc.
        """
        # Extract ADALINE-specific configuration
        adaline_config = {
            "model_type": "ADALINE",
            "algorithm": "Adaptive Linear Neuron",
            "learning_rule": "Delta Rule (LMS)",
            "activation": "Linear (No Activation)",
            "loss_function": "Mean Squared Error",
        }
        # Only include scalar config values (int, float, str, bool)
        for k, v in config.items():
            if isinstance(v, (int, float, str, bool)):
                adaline_config[k] = v
            else:
                # Optionally, convert to string or skip
                continue
        if self.enabled:
            # Use allow_val_change=True to allow config updates
            self.wandb_run.config.update(adaline_config, allow_val_change=True)
            logger.info(f"Logged ADALINE configuration: {list(adaline_config.keys())}")
    
    def log_training_progress(self, metrics: Dict[str, Any], step: int) -> None:
        """Log training progress metrics (implements abstract method).
        
        Args:
            metrics: Training metrics including loss, weight updates
            step: Current training epoch/iteration
        """
        if self.enabled:
            self.wandb_run.log(metrics, step=step)
            logger.debug(f"Logged training progress at step {step}: {list(metrics.keys())}")
    
    def create_model_visualizations(self, **kwargs) -> None:
        """Create ADALINE-specific visualizations (implements abstract method).
        
        Args:
            **kwargs: Visualization parameters including model, data, predictions
        """
        model = kwargs.get('model')
        X = kwargs.get('X')
        y = kwargs.get('y')
        training_state = kwargs.get('training_state')
        
        if model is not None and training_state is not None:
            # Create training progress visualization
            self._log_training_progress_plot(training_state)
        
        if model is not None and training_state is not None:
            # Create weight evolution visualization
            self._log_weight_evolution_plot(training_state)
        
        if model is not None and X is not None and y is not None:
            # Create decision boundary visualization (for 2D data)
            self._log_decision_boundary(model, X, y)
        
        if model is not None and X is not None and y is not None:
            # Create model comparison visualization
            self._log_model_comparison(model, X, y)
    
    def log_final_metrics_summary(self, metrics: Dict[str, float]) -> None:
        """Log final evaluation metrics as W&B summary values."""
        if self.enabled:
            for key, value in metrics.items():
                self.wandb_run.summary[key] = value
            logger.info(f"Logged final metrics to W&B summary: {list(metrics.keys())}")

    def log_training_results(self, model: Any, training_state: Any, 
                           X: np.ndarray, y: np.ndarray, 
                           predictions: np.ndarray, metrics: Dict[str, float]) -> None:
        """Comprehensive logging of training results and visualizations.
        
        This method logs:
        - Model configuration
        - Evaluation metrics
        - All major plots (training progress, weight evolution, model comparison, decision boundary) as images to W&B
        
        Args:
            model: Trained ADALINE model
            training_state: Training state object containing history
            X: Input features
            y: True targets
            predictions: Model predictions
            metrics: Evaluation metrics
        """
        try:
            if not self.enabled:
                logger.info("Visualization disabled - skipping training results logging")
                return
            
            logger.info("Logging comprehensive ADALINE training results...")
            
            # Log model configuration
            model_config = {
                "learning_rate": getattr(model, 'learning_rate', 'unknown'),
                "input_size": getattr(model, 'input_size', 'unknown'),
                "convergence_epoch": getattr(training_state, 'convergence_epoch', 'unknown'),
                "final_loss": getattr(training_state, 'final_loss', 'unknown'),
                "total_epochs": len(getattr(training_state, 'training_loss', []))
            }
            self.log_model_config(model_config)
            
            # Log evaluation metrics
            self._log_evaluation_metrics(metrics)
            # Log final metrics as summary
            self.log_final_metrics_summary(metrics)
            
            # Log all major visualizations as images
            self.create_model_visualizations(
                model=model,
                X=X,
                y=y,
                training_state=training_state
            )
            
            # Log weight analysis
            self._log_weight_analysis(model, training_state)
            
            logger.info("ADALINE training results logging complete")
            
        except Exception as e:
            logger.error(f"Failed to log training results: {e}")

    def _log_training_progress_plot(self, training_state: Any) -> None:
        """Create and log training progress visualization."""
        try:
            if not self.enabled or not hasattr(training_state, 'training_loss'):
                return
            
            # Create training progress plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('ADALINE Training Progress', fontsize=16, fontweight='bold')
            
            epochs = np.arange(1, len(training_state.training_loss) + 1)
            
            # Training loss
            axes[0, 0].plot(epochs, training_state.training_loss, 
                           color='#1f77b4', linewidth=2, label='Training Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Validation loss (if available)
            if hasattr(training_state, 'validation_loss') and len(training_state.validation_loss) > 0:
                axes[0, 1].plot(epochs, training_state.validation_loss, 
                               color='#ff7f0e', linewidth=2, label='Validation Loss')
                axes[0, 1].set_title('Validation Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()
            
            # Loss comparison
            if hasattr(training_state, 'validation_loss') and len(training_state.validation_loss) > 0:
                axes[1, 0].plot(epochs, training_state.training_loss, 
                               color='#1f77b4', linewidth=2, label='Training')
                axes[1, 0].plot(epochs, training_state.validation_loss, 
                               color='#ff7f0e', linewidth=2, label='Validation')
                axes[1, 0].set_title('Training vs Validation Loss')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
            
            # Loss improvement
            if len(training_state.training_loss) > 1:
                loss_improvement = np.diff(training_state.training_loss)
                axes[1, 1].plot(epochs[1:], loss_improvement, 
                               color='#2ca02c', linewidth=2)
                axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
                axes[1, 1].set_title('Loss Improvement per Epoch')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Loss Change')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.log_figure(fig, "training_progress")
            
        except Exception as e:
            logger.warning(f"Could not create training progress plot: {e}")
    
    def _log_weight_evolution_plot(self, training_state: Any) -> None:
        """Create and log weight evolution visualization."""
        try:
            if not self.enabled or not hasattr(training_state, 'weight_history'):
                return
            
            weight_history = training_state.weight_history
            bias_history = training_state.bias_history
            
            n_weights = weight_history.shape[1]
            epochs = np.arange(1, len(weight_history) + 1)
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle('ADALINE Weight Evolution', fontsize=16, fontweight='bold')
            
            # Weight evolution
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            for i in range(n_weights):
                axes[0].plot(epochs, weight_history[:, i], 
                            color=colors[i % len(colors)], 
                            linewidth=2, label=f'Weight {i+1}')
            
            axes[0].set_title('Weight Evolution')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Weight Value')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Bias evolution
            axes[1].plot(epochs, bias_history, 
                        color=colors[-1], linewidth=2, label='Bias')
            axes[1].set_title('Bias Evolution')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Bias Value')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            
            plt.tight_layout()
            self.log_figure(fig, "weight_evolution")
            
        except Exception as e:
            logger.warning(f"Could not create weight evolution plot: {e}")
    
    def _log_decision_boundary(self, model: Any, X: np.ndarray, y: np.ndarray) -> None:
        """Create and log decision boundary visualization."""
        try:
            if not self.enabled:
                return
            
            # Check if we can create a decision boundary (need 2D features)
            # If X has bias term (3 features), we need to extract the original 2 features
            if X.shape[1] == 3:  # Has bias term
                # Extract original features (skip bias term)
                X_2d = X[:, 1:3]  # Skip the bias column
            elif X.shape[1] == 2:  # Already 2D
                X_2d = X
            else:
                logger.info("Decision boundary plot only works for 2D data")
                return
            
            # Create meshgrid for decision boundary
            x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
            y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                 np.linspace(y_min, y_max, 100))
            
            # Make predictions on meshgrid
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            # Add bias term if needed
            if X.shape[1] == 3:  # Original data has bias term
                grid_points = np.hstack([np.ones((grid_points.shape[0], 1)), grid_points])
            
            Z = model.predict(grid_points)
            Z = Z.reshape(xx.shape)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('ADALINE Decision Boundary', fontsize=16, fontweight='bold')
            
            # Training data
            scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y.flatten(), 
                                  cmap='viridis', alpha=0.7, s=50)
            contour1 = ax1.contour(xx, yy, Z, levels=[0], colors='red', linewidths=2)
            ax1.set_title('Training Data with Decision Boundary')
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')
            ax1.grid(True, alpha=0.3)
            
            # Test data (if available)
            ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=y.flatten(), 
                       cmap='viridis', alpha=0.7, s=50)
            ax2.contour(xx, yy, Z, levels=[0], colors='red', linewidths=2)
            ax2.set_title('Test Data with Decision Boundary')
            ax2.set_xlabel('Feature 1')
            ax2.set_ylabel('Feature 2')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.log_figure(fig, "decision_boundary")
            
        except Exception as e:
            logger.warning(f"Could not create decision boundary plot: {e}")
    
    def _log_model_comparison(self, model: Any, X: np.ndarray, y: np.ndarray) -> None:
        """Create and log model comparison visualization."""
        try:
            if not self.enabled:
                return
            
            # Make predictions
            predictions = model.predict(X)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('ADALINE Model Analysis', fontsize=16, fontweight='bold')
            
            # Actual vs Predicted
            axes[0, 0].scatter(y.flatten(), predictions, alpha=0.6)
            axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title('Actual vs Predicted')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Residuals
            residuals = y.flatten() - predictions
            axes[0, 1].scatter(predictions, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted Values')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residual Plot')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Residual histogram
            axes[1, 0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Residual Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Residual Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Q-Q plot of residuals
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot of Residuals')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.log_figure(fig, "model_comparison")
            
        except Exception as e:
            logger.warning(f"Could not create model comparison plot: {e}")
    
    def _log_weight_analysis(self, model: Any, training_state: Any) -> None:
        """Log weight analysis and evolution."""
        try:
            if not self.enabled or not hasattr(model, 'weights'):
                return
            
            # Log final weights
            weight_metrics = {
                "final_weight_norm": float(np.linalg.norm(model.weights)),
                "final_bias": float(model.bias) if hasattr(model, 'bias') else 0.0,
                "weight_magnitude_avg": float(np.mean(np.abs(model.weights))),
                "weight_magnitude_max": float(np.max(np.abs(model.weights))),
                "weight_magnitude_min": float(np.min(np.abs(model.weights))),
                "weight_std": float(np.std(model.weights))
            }
            
            self.log_metrics(weight_metrics)
            
            # Create weight histogram if we have weight history
            if hasattr(training_state, 'weight_history') and training_state.weight_history is not None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Weight evolution over time
                weight_history = training_state.weight_history
                epochs = range(len(weight_history))
                
                for i in range(weight_history.shape[1]):
                    ax1.plot(epochs, weight_history[:, i], 
                            label=f'Weight {i+1}', marker='o', markersize=3)
                
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Weight Value')
                ax1.set_title('Weight Evolution During Training')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Final weight distribution
                ax2.hist(model.weights, bins=10, alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Weight Value')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Final Weight Distribution')
                ax2.grid(True, alpha=0.3)
                
                self.log_figure(fig, "weight_analysis")
                
        except Exception as e:
            logger.warning(f"Could not create weight analysis: {e}")
    
    def _log_evaluation_metrics(self, metrics: Dict[str, float]) -> None:
        """Log evaluation metrics."""
        try:
            if not self.enabled:
                return
            
            # Log all metrics
            self.log_metrics(metrics)
            
            # Create metrics summary
            metrics_summary = {
                "model_performance/mse": metrics.get('mse', 0.0),
                "model_performance/mae": metrics.get('mae', 0.0),
                "model_performance/r2_score": metrics.get('r2_score', 0.0),
                "model_performance/rmse": metrics.get('rmse', 0.0),
                "model_performance/mape": metrics.get('mape', 0.0)
            }
            
            self.log_metrics(metrics_summary)
            
        except Exception as e:
            logger.warning(f"Could not log evaluation metrics: {e}")


def initialize_wandb(project_name: str = "ai-from-scratch-adaline",
                    entity: Optional[str] = None,
                    config: Optional[Dict[str, Any]] = None,
                    enabled: bool = True) -> tuple:
    """Initialize Weights & Biases for ADALINE experiments.
    
    Args:
        project_name: W&B project name
        entity: W&B entity (username or team)
        config: Initial configuration to log
        enabled: Whether to enable W&B logging
        
    Returns:
        Tuple of (wandb_run, visualizer)
    """
    if not enabled:
        return None, ADALINEWandbVisualizer(enabled=False)
    
    try:
        import wandb
        
        # Initialize W&B run
        wandb_run = wandb.init(
            project=project_name,
            entity=entity,
            config=config or {},
            name=f"adaline-experiment-{wandb.util.generate_id()}",
            tags=["adaline", "regression", "delta-rule"]
        )
        
        # Create visualizer
        visualizer = ADALINEWandbVisualizer(wandb_run, enabled=True)
        
        logger.info(f"W&B initialized for ADALINE experiments: {project_name}")
        return wandb_run, visualizer
        
    except ImportError:
        logger.warning("Weights & Biases not available. Install with: pip install wandb")
        return None, ADALINEWandbVisualizer(enabled=False)
    except Exception as e:
        logger.error(f"Failed to initialize W&B: {e}")
        return None, ADALINEWandbVisualizer(enabled=False)


def finish_wandb(wandb_run: Optional[Any]) -> None:
    """Finish W&B run and cleanup.
    
    Args:
        wandb_run: Active W&B run to finish
    """
    if wandb_run is not None:
        try:
            wandb_run.finish()
            logger.info("W&B run finished successfully")
        except Exception as e:
            logger.warning(f"Failed to finish W&B run: {e}")


# Backward compatibility alias
Visualizer = ADALINEWandbVisualizer 