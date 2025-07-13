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
- Leverage updated visualization framework with Phase 2 and Phase 3 features
"""

from typing import Dict, Any, Optional, List
import numpy as np
import logging
import sys
import os
import matplotlib.pyplot as plt

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import from standardized shared package with fallback
try:
    from ai_from_scratch_shared.visualization import (
        BaseVisualizer,
        InteractiveVisualizer,
        AdvancedVisualizer
    )
    from ai_from_scratch_shared.wandb_integration import BaseWandbVisualizer
    SHARED_FRAMEWORK_AVAILABLE = True
except ImportError:
    # Fallback for development/testing
    print("Warning: Could not import shared visualization framework")
    BaseVisualizer = object
    BaseWandbVisualizer = object
    InteractiveVisualizer = object
    AdvancedVisualizer = object
    SHARED_FRAMEWORK_AVAILABLE = False

# Import the updated Perceptron visualizer
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
    - Interactive visualizations (Phase 2)
    - Advanced analysis (Phase 2)
    """
    
    def __init__(self, wandb_run: Optional[Any] = None, enabled: bool = True) -> None:
        """Initialize the Perceptron W&B visualizer.
        
        Args:
            wandb_run: Active Weights & Biases run object
            enabled: Whether to enable W&B logging
        """
        if SHARED_FRAMEWORK_AVAILABLE:
            super().__init__(wandb_run, enabled)
        else:
            # Fallback initialization
            self.wandb_run = wandb_run
            self.enabled = enabled
        
        # Initialize the updated Perceptron visualizer
        self.visualizer = PerceptronVisualizer(enabled=enabled)
        
        # Initialize Phase 2 features if available
        if SHARED_FRAMEWORK_AVAILABLE:
            self.interactive_viz = InteractiveVisualizer(model_name="Perceptron")
            self.advanced_viz = AdvancedVisualizer(model_name="Perceptron")
        
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
            "framework_version": "Updated with Phase 2 & 3 features",
            **config  # Include all provided configuration
        }
        
        if self.enabled and hasattr(self, 'wandb_run') and self.wandb_run:
            self.wandb_run.config.update(perceptron_config)
            logger.info(f"Logged perceptron configuration: {list(perceptron_config.keys())}")
    
    def log_training_progress(self, metrics: Dict[str, Any], step: int) -> None:
        """Log training progress metrics (implements abstract method).
        
        Args:
            metrics: Training metrics including loss, accuracy, weight updates
            step: Current training epoch/iteration
        """
        if self.enabled and hasattr(self, 'wandb_run') and self.wandb_run:
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
            
        # Phase 2: Interactive and Advanced visualizations
        if SHARED_FRAMEWORK_AVAILABLE:
            self._log_interactive_visualizations(model, X, y)
            self._log_advanced_visualizations(model, X, y)
    
    def log_training_results(self, model, X: np.ndarray, y: np.ndarray, 
                           predictions: np.ndarray, class_names: Optional[List[str]] = None,
                           experiment_name: Optional[str] = None) -> None:
        """Comprehensive logging of training results and visualizations.
        
        Args:
            model: Trained perceptron model
            X: Input features
            y: True labels
            predictions: Model predictions
            class_names: Optional class names for labeling
            experiment_name: Name of the experiment for axis labels
        """
        import numpy as np
        import matplotlib.pyplot as plt
        try:
            if not self.enabled:
                logger.info("Visualization disabled - skipping training results logging")
                return
            
            logger.info("Logging comprehensive training results with updated framework...")
            
            # Log model configuration
            model_config = {
                "learning_rate": getattr(model, 'learning_rate', 0.01),
                "n_iterations": getattr(model, 'n_iters', 100),
                "input_features": X.shape[1],
                "n_samples": X.shape[0],
                "converged": getattr(model, 'converged_', False),
                "framework_features": ["Phase 1: Base Framework", "Phase 2: Interactive & Advanced", "Phase 3: Performance & Testing"]
            }
            self.log_model_config(model_config)
            
            # Log classification metrics
            self._log_classification_metrics(y, predictions)
            
            # Get run name for unique subfolder
            run_name = None
            if hasattr(self, 'wandb_run') and self.wandb_run:
                run_name = getattr(self.wandb_run, 'name', None)
            run_subdir = f"outputs/plots/{run_name}" if run_name else "outputs/plots/unknown_run"
            import os
            os.makedirs(run_subdir, exist_ok=True)

            # Generate comprehensive visualizations using updated framework
            errors_per_epoch = getattr(model, 'errors_per_epoch', [])
            run_visualizer = self.visualizer.__class__(save_dir=run_subdir, enabled=self.enabled)
            visualizations = run_visualizer.generate_all_visualizations(
                model, X, y, predictions, errors_per_epoch, class_names, experiment_name=experiment_name, run_subdir=run_subdir
            )
            
            # Log all generated visualizations to W&B
            for viz_name, fig in visualizations.items():
                if fig is not None:
                    self.log_figure(fig, viz_name)
                    plt.close(fig)

            # Log weights evolution GIF if available
            weights_history = getattr(model, 'weights_history', None)
            bias_history = getattr(model, 'bias_history', None)
            accuracy_history = getattr(model, 'accuracy_history', None)
            if weights_history is not None:
                weights_history = np.array(weights_history)
                if weights_history.ndim == 2 and weights_history.shape[1] == 784:
                    gif_path = run_visualizer.create_weights_animation_gif(
                        weights_history,
                        save_path=os.path.join(run_subdir, "perceptron_weights_evolution.gif"),
                        experiment_name=experiment_name or "mnist"
                    )
                    if hasattr(self, 'wandb_run') and self.wandb_run:
                        import wandb
                        self.wandb_run.log({"Weights_Evolution_GIF": wandb.Video(gif_path, format="gif")})
            # Log weights distribution plot
            if weights_history is not None:
                fig = run_visualizer.plot_weights_distribution(weights_history, save_name="weights_dist.png", experiment_name=experiment_name or "mnist")
                if hasattr(self, 'wandb_run') and self.wandb_run:
                    self.log_figure(fig, "Weights_Dist")
                plt.close(fig)
            # Log bias evolution plot
            if bias_history is not None:
                fig = run_visualizer.plot_bias_evolution(np.array(bias_history), save_name="bias_dist.png", experiment_name=experiment_name or "mnist")
                if hasattr(self, 'wandb_run') and self.wandb_run:
                    self.log_figure(fig, "Bias_Dist")
                plt.close(fig)
            # Log accuracy evolution plot
            if accuracy_history is not None:
                fig = run_visualizer.plot_accuracy_evolution(np.array(accuracy_history), save_name="accuracy_evolution.png", experiment_name=experiment_name or "mnist")
                if hasattr(self, 'wandb_run') and self.wandb_run:
                    self.log_figure(fig, "Accuracy_Evolution")
                plt.close(fig)
            
            # Log minimal 28x28 weights evolution GIF
            if weights_history is not None:
                minimal_gif_path = run_visualizer.create_minimal_weights_gif(weights_history, save_path=os.path.join(run_subdir, "perceptron_weights_minimal.gif"))
                if hasattr(self, 'wandb_run') and self.wandb_run:
                    import wandb
                    self.wandb_run.log({"Weights_Evolution_Minimal_GIF": wandb.Video(minimal_gif_path, format="gif")})
            # Log final weights heatmap
            if weights_history is not None:
                final_weights = weights_history[-1]
                final_weights_path = run_visualizer.plot_final_weights_distribution(final_weights, save_path=os.path.join(run_subdir, "perceptron_final_weights.png"), experiment_name=experiment_name or "mnist")
                if hasattr(self, 'wandb_run') and self.wandb_run:
                    self.wandb_run.log({"Final_Weights_Heatmap": wandb.Image(final_weights_path)})
            
            # Log confusion matrix as W&B Table and image
            if hasattr(self, 'wandb_run') and self.wandb_run:
                import wandb
                from sklearn.metrics import confusion_matrix
                import numpy as np
                y_true = y
                y_pred = predictions
                cm = confusion_matrix(y_true, y_pred)
                # Log as Table
                cm_table = wandb.Table(columns=["Predicted_0", "Predicted_1"], data=cm.tolist())
                self.wandb_run.log({"Confusion_Matrix_Table": cm_table})
                # Log as image
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                im = ax.imshow(cm, cmap="Blues")
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")
                ax.set_title("Confusion Matrix")
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
                fig.tight_layout()
                self.wandb_run.log({"Confusion_Matrix_Image": wandb.Image(fig)})
                plt.close(fig)
                # Log class distribution as histogram
                unique, counts = np.unique(y_true, return_counts=True)
                class_dist = dict(zip([str(u) for u in unique], counts.tolist()))
                self.wandb_run.log({"Class_Distribution": wandb.Histogram(list(y_true))})
                # Log interactive confusion matrix chart
                try:
                    class_names_for_chart = class_names if class_names is not None else ["0", "1"]
                    self.wandb_run.log({
                        "Confusion_Matrix_Chart": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=list(y_true),
                            preds=list(y_pred),
                            class_names=class_names_for_chart
                        )
                    })
                except Exception as e:
                    print(f"Failed to log interactive confusion matrix chart: {e}")
            
            logger.info(f"Training results logging complete - {len(visualizations)} visualizations generated")
            
        except Exception as e:
            import logging
            logging.error(f"Failed to log training results: {e}")

    def _log_decision_boundary(self, model: Any, X: np.ndarray, y: np.ndarray) -> None:
        """Create and log decision boundary visualization using updated framework."""
        try:
            if not self.enabled or X.shape[1] != 2:
                return  # Only create 2D decision boundaries
            
            # Use the updated framework visualizer
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
        """Create and log learning curve visualization using updated framework."""
        try:
            if not self.enabled or not losses:
                return
            
            # Use the updated framework visualizer
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
            if not self.enabled:
                return
                
            # Extract weight information if available
            weights = getattr(model, 'weights', None)
            bias = getattr(model, 'bias', 0)
            
            if weights is not None:
                weight_data = {
                    "weight_0": float(weights[0]) if len(weights) > 0 else 0,
                    "weight_1": float(weights[1]) if len(weights) > 1 else 0,
                    "bias": float(bias),
                    "weight_magnitude": float(np.linalg.norm(weights)),
                    "num_weights": len(weights)
                }
                
                if hasattr(self, 'wandb_run') and self.wandb_run:
                    self.wandb_run.log(weight_data)
                    logger.debug(f"Logged weight analysis: {list(weight_data.keys())}")
                    
        except Exception as e:
            logger.warning(f"Could not log weight analysis: {e}")
    
    def _log_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Log comprehensive classification metrics."""
        try:
            if not self.enabled:
                return
                
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division='warn')
            recall = recall_score(y_true, y_pred, zero_division='warn')
            f1 = f1_score(y_true, y_pred, zero_division='warn')
            
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            
            if hasattr(self, 'wandb_run') and self.wandb_run:
                self.wandb_run.log(metrics)
                logger.info(f"Logged classification metrics: accuracy={accuracy:.3f}, precision={precision:.3f}, recall={recall:.3f}")
                
        except Exception as e:
            logger.warning(f"Could not log classification metrics: {e}")
    
    def _log_interactive_visualizations(self, model: Any, X: np.ndarray, y: np.ndarray) -> None:
        """Log interactive visualizations using Phase 2 features."""
        try:
            if not self.enabled or not SHARED_FRAMEWORK_AVAILABLE or X.shape[1] != 2:
                return
                
            # Create interactive visualization
            fig = self.visualizer.create_interactive_visualization(
                model, X, y, "Interactive Perceptron Learning Demo"
            )
            
            if fig is not None:
                self.log_figure(fig, "interactive_demo")
                plt.close(fig)
                logger.info("Logged interactive visualization")
                
        except Exception as e:
            logger.warning(f"Could not create interactive visualization: {e}")
    
    def _log_advanced_visualizations(self, model: Any, X: np.ndarray, y: np.ndarray) -> None:
        """Log advanced visualizations using Phase 2 features."""
        try:
            if not self.enabled or not SHARED_FRAMEWORK_AVAILABLE:
                return
                
            # Create advanced visualization
            fig = self.visualizer.create_advanced_visualization(
                model, X, y, "feature_importance", "Advanced Perceptron Analysis"
            )
            
            if fig is not None:
                self.log_figure(fig, "advanced_analysis")
                plt.close(fig)
                logger.info("Logged advanced visualization")
                
        except Exception as e:
            logger.warning(f"Could not create advanced visualization: {e}")
    
    def log_figure(self, fig: plt.Figure, name: str) -> None:
        """Log a matplotlib figure to W&B.
        
        Args:
            fig: Matplotlib figure to log
            name: Name for the figure in W&B
        """
        try:
            if self.enabled and hasattr(self, 'wandb_run') and self.wandb_run:
                import wandb
                self.wandb_run.log({name: wandb.Image(fig)})
                logger.debug(f"Logged figure: {name}")
        except Exception as e:
            logger.warning(f"Could not log figure {name}: {e}")


# Export for backward compatibility
__all__ = ['PerceptronWandbVisualizer']
