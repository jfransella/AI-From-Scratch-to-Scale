"""
Perceptron Visualizer using Updated Shared Framework
==================================================

This module provides Perceptron specific visualizations using the updated shared
visualization framework. It demonstrates how to extend the BaseVisualizer for
binary classification tasks while leveraging all the new Phase 2 and Phase 3 features.

Key Features:
- Confusion matrix with binary classification insights
- Learning curve tracking convergence behavior
- Decision boundary visualization for 2D data
- Educational annotations about linear separability
- Professional styling consistent across models
- Interactive visualizations (Phase 2)
- Advanced plot types (Phase 2)
- Performance optimization (Phase 3)

Educational Focus:
- Linear separability concepts
- Perceptron convergence theorem
- Binary classification fundamentals
- Decision boundary interpretation
- Historical context of neural networks
- Interactive learning experiences
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.figure import Figure
import matplotlib.animation as animation

# Import updated shared visualization framework
try:
    from ai_from_scratch_shared.visualization import (
        BaseVisualizer,
        PlotFactory,
        InteractiveVisualizer,
        AdvancedVisualizer,
        ConfusionMatrixVisualizer,
        TrainingCurveVisualizer,
        DecisionBoundaryVisualizer,
        EducationalAnnotator,
        add_mathematical_context,
        add_performance_insights,
        create_concept_explanation,
        apply_educational_theme,
        get_model_color_scheme
    )
    SHARED_FRAMEWORK_AVAILABLE = True
except ImportError:
    # Fallback for development/testing
    print("Warning: Could not import shared visualization framework")
    SHARED_FRAMEWORK_AVAILABLE = False

# Handle both relative and absolute imports for constants
try:
    from .constants import DECISION_BOUNDARY_RESOLUTION
except ImportError:
    from constants import DECISION_BOUNDARY_RESOLUTION

logger = logging.getLogger(__name__)


class PerceptronVisualizer(BaseVisualizer):
    """Perceptron-specific visualizer extending the updated shared framework.

    Provides comprehensive visualization capabilities for Perceptron models,
    focusing on binary classification, linear separability, and convergence
    behavior with educational context and professional styling.
    
    Now includes Phase 2 interactive features and Phase 3 performance optimizations.
    """

    def __init__(self, save_dir: Union[str, Path] = "outputs/plots", enabled: bool = True):
        """Initialize the Perceptron visualizer with updated framework features.

        Args:
            save_dir: Directory to save visualization files
            enabled: Whether to enable visualization generation
        """
        if not SHARED_FRAMEWORK_AVAILABLE:
            raise ImportError("Shared visualization framework is required but not available")
            
        super().__init__(model_name="Perceptron", default_save_dir=save_dir)
        self.enabled = enabled

        # Initialize specialized visualizers from shared framework
        self.plot_factory = PlotFactory(model_name="Perceptron")
        self.confusion_matrix_viz = ConfusionMatrixVisualizer()
        self.training_curve_viz = TrainingCurveVisualizer()
        self.decision_boundary_viz = DecisionBoundaryVisualizer()
        self.educational_annotator = EducationalAnnotator()
        
        # Initialize Phase 2 features
        self.interactive_viz = InteractiveVisualizer(model_name="Perceptron")
        self.advanced_viz = AdvancedVisualizer(model_name="Perceptron")

        # Perceptron-specific color scheme
        self.perceptron_colors = get_model_color_scheme("Perceptron")

        # Apply educational theme
        apply_educational_theme()

        logger.info("PerceptronVisualizer initialized with updated framework (enabled: %s)", enabled)

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             title: str = "Perceptron Confusion Matrix",
                             save_name: Optional[str] = "confusion_matrix",
                             xlabel: str = "Predicted Label",
                             ylabel: str = "True Label") -> Optional[Figure]:
        """Create enhanced confusion matrix with binary classification insights.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names for the classes
            title: Plot title
            save_name: Filename for saving
            xlabel: Label for the x-axis
            ylabel: Label for the y-axis

        Returns:
            matplotlib Figure object or None if disabled
        """
        if not self.enabled:
            return None
        logger.info("📊 Generating enhanced confusion matrix...")
        
        if class_names is None:
            class_names = ['Class 0', 'Class 1']
            
        # Use shared framework confusion matrix visualizer
        fig, ax = self.confusion_matrix_viz.plot(
            y_true, y_pred, class_names, title=title, xlabel=xlabel, ylabel=ylabel
        )
        
        if fig is not None:
            # Removed annotation/educational overlays for a cleaner confusion matrix
            # (No calls to add_mathematical_context, add_performance_insights, or create_concept_explanation)
            if save_name:
                self.save_and_show(fig, save_name)
                
        return fig

    def plot_learning_curve(self, errors_per_epoch: List[int],
                          title: str = "Perceptron Learning Curve",
                          save_name: Optional[str] = "learning_curve") -> Optional[Figure]:
        """Create enhanced learning curve showing convergence behavior.

        Args:
            errors_per_epoch: List of error counts per epoch
            title: Plot title
            save_name: Filename for saving

        Returns:
            matplotlib Figure object or None if disabled
        """
        if not self.enabled:
            return None
        logger.info("📈 Generating enhanced learning curve...")
        
        # Use shared framework training curve visualizer
        fig, ax = self.training_curve_viz.plot_learning_curve(
            errors_per_epoch, title
        )
        
        if fig is not None:
            # Add Perceptron-specific educational context
            add_mathematical_context(
                ax,
                "Convergence Theorem",
                "Perceptron converges in finite steps for linearly separable data",
                "Error count should decrease to zero for separable data"
            )
            
            # Add convergence analysis
            if errors_per_epoch:
                final_errors = errors_per_epoch[-1]
                convergence_status = 1.0 if final_errors == 0 else 0.0
                epochs_to_converge = len(errors_per_epoch) if final_errors == 0 else 0
                
                add_performance_insights(
                    ax,
                    metrics={"Convergence": convergence_status, "Final Errors": float(final_errors)},
                    interpretations={
                        "Convergence": "1.0 = Converged, 0.0 = Not Converged",
                        "Final Errors": "Errors at final epoch"
                    }
                )
            
            if save_name:
                self.save_and_show(fig, save_name)
                
        return fig

    def plot_decision_boundary(self, model, features: np.ndarray, y: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             title: str = "Perceptron Decision Boundary",
                             save_name: Optional[str] = "decision_boundary") -> Optional[Figure]:
        """Create decision boundary visualization for 2D data.

        Args:
            model: Trained perceptron model
            features: Input features (2D)
            y: True labels
            class_names: Names for the classes
            title: Plot title
            save_name: Filename for saving

        Returns:
            matplotlib Figure object or None if disabled
        """
        if not self.enabled or features.shape[1] != 2:
            return None
            
        logger.info("🎯 Generating decision boundary visualization...")
        
        if class_names is None:
            class_names = ['Class 0', 'Class 1']
            
        # Use shared framework decision boundary visualizer
        fig, ax = self.decision_boundary_viz.plot(
            model, features, y, class_names, title=title
        )
        
        if fig is not None:
            # Add Perceptron-specific educational context
            add_mathematical_context(
                ax,
                "Decision Boundary",
                "w₁x₁ + w₂x₂ + b = 0",
                "Linear separation of classes"
            )
            
            # Add concept explanation
            create_concept_explanation(
                ax,
                "Linear Decision Boundary",
                "The Perceptron creates a linear decision boundary that separates "
                "the two classes. This boundary is defined by the learned weights."
            )
            
            if save_name:
                self.save_and_show(fig, save_name)
                
        return fig

    def create_interactive_visualization(self, model, features: np.ndarray, y: np.ndarray,
                                       title: str = "Interactive Perceptron Demo") -> Optional[Figure]:
        """Create interactive visualization using Phase 2 features.

        Args:
            model: Trained perceptron model
            features: Input features
            y: True labels
            title: Plot title

        Returns:
            matplotlib Figure object or None if disabled
        """
        if not self.enabled:
            return None
            
        logger.info("🎮 Generating interactive visualization...")
        
        # Use Phase 2 interactive features
        fig, ax = self.interactive_viz.create_interactive_decision_boundary(
            model, features, y
        )
        
        return fig

    def create_advanced_visualization(self, model, features: np.ndarray, y: np.ndarray,
                                    plot_type: str = "gradient_flow",
                                    title: str = "Advanced Perceptron Analysis") -> Optional[Figure]:
        """Create advanced visualization using Phase 2 features.

        Args:
            model: Trained perceptron model
            features: Input features
            y: True labels
            plot_type: Type of advanced plot
            title: Plot title

        Returns:
            matplotlib Figure object or None if disabled
        """
        if not self.enabled:
            return None
            
        logger.info("🔬 Generating advanced visualization...")
        
        # Use Phase 2 advanced features
        if plot_type == "gradient_flow":
            fig, ax = self.advanced_viz.create_gradient_flow(
                gradients=[], layer_names=[]
            )
        else:
            fig, ax = self.advanced_viz.create_feature_importance(
                feature_names=[], importance_scores=np.array([])
            )
        
        return fig

    def generate_all_visualizations(self, model, features: np.ndarray, y: np.ndarray,
                                  y_pred: np.ndarray, errors_per_epoch: List[int],
                                  class_names: Optional[List[str]] = None,
                                  experiment_name: Optional[str] = None) -> Dict[str, Figure]:
        """Generate all visualizations for the perceptron model.

        Args:
            model: Trained perceptron model
            features: Input features
            y: True labels
            y_pred: Model predictions
            errors_per_epoch: List of error counts per epoch
            class_names: Names for the classes
            experiment_name: Name of the experiment for axis labels

        Returns:
            Dictionary mapping visualization names to Figure objects
        """
        if not self.enabled:
            return {}
            
        logger.info("🎨 Generating comprehensive visualization suite...")
        
        visualizations = {}
        
        # Determine axis labels based on experiment
        if experiment_name is None:
            experiment_name = ""
        experiment_name = experiment_name.lower() if experiment_name else ""
        if "mnist" in experiment_name:
            xlabel, ylabel = "Predicted Digit", "True Digit"
        elif "iris" in experiment_name:
            xlabel, ylabel = "Predicted Species", "True Species"
        elif experiment_name in ["and", "xor"]:
            xlabel, ylabel = "Predicted Output", "True Output"
        else:
            xlabel, ylabel = "Predicted Label", "True Label"

        # Standard visualizations
        if y is not None and y_pred is not None:
            visualizations['confusion_matrix'] = self.plot_confusion_matrix(
                y, y_pred, class_names, xlabel=xlabel, ylabel=ylabel
            )
            
        if errors_per_epoch:
            visualizations['learning_curve'] = self.plot_learning_curve(errors_per_epoch)
            
        if features.shape[1] == 2:
            visualizations['decision_boundary'] = self.plot_decision_boundary(
                model, features, y, class_names
            )
            
        # Phase 2: Interactive and Advanced visualizations
        if features.shape[1] == 2:  # Only for 2D data
            visualizations['interactive_demo'] = self.create_interactive_visualization(
                model, features, y
            )
            
        visualizations['advanced_analysis'] = self.create_advanced_visualization(
            model, features, y, plot_type="feature_importance"
        )
        
        logger.info(f"Generated {len(visualizations)} visualizations")
        return visualizations

    def create_weights_animation_gif(self, weight_history: np.ndarray, save_path: str = "outputs/plots/perceptron_weights_evolution.gif", experiment_name: str = "mnist") -> str:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import numpy as np
        import os

        if weight_history.ndim != 2 or weight_history.shape[1] != 784:
            raise ValueError("weight_history must be of shape (epochs, 784)")
        n_epochs = weight_history.shape[0]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        vmin, vmax = np.min(weight_history), np.max(weight_history)
        im = ax.imshow(weight_history[0].reshape(28, 28), cmap="coolwarm", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Weight Value")
        ax.set_title(f"Perceptron Weights Evolution ({experiment_name.upper()})")
        epoch_text = ax.text(0, 1, f"Epoch 1/{n_epochs}", color="black", fontsize=12, va="top", ha="left", bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.axis("off")

        def update(epoch):
            im.set_data(weight_history[epoch].reshape(28, 28))
            epoch_text.set_text(f"Epoch {epoch+1}/{n_epochs}")
            return [im, epoch_text]

        anim = FuncAnimation(fig, update, frames=n_epochs, interval=400, blit=False)
        anim.save(save_path, writer='pillow')
        plt.close(fig)
        return save_path

    def plot_weights_distribution(self, weights_history: np.ndarray, save_name: str = "weights_dist", experiment_name: str = "mnist"):
        import numpy as np
        import matplotlib.pyplot as plt
        if weights_history.ndim != 2:
            weights_history = np.array(weights_history)
        epochs = np.arange(1, weights_history.shape[0] + 1)
        mean_w = np.mean(weights_history, axis=1)
        min_w = np.min(weights_history, axis=1)
        max_w = np.max(weights_history, axis=1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, mean_w, label="Mean Weight", color="blue")
        ax.plot(epochs, min_w, label="Min Weight", color="red", linestyle="--")
        ax.plot(epochs, max_w, label="Max Weight", color="green", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weight Value")
        if "mnist" in experiment_name.lower():
            ax.set_title("Perceptron Weights Distribution (MNIST)")
        else:
            ax.set_title(f"Perceptron Weights Distribution ({experiment_name})")
        ax.legend()
        self.save_and_show(fig, save_name)
        return fig

    def plot_bias_evolution(self, bias_history: np.ndarray, save_name: str = "bias_dist", experiment_name: str = "mnist"):
        import numpy as np
        import matplotlib.pyplot as plt
        epochs = np.arange(1, len(bias_history) + 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, bias_history, label="Bias", color="purple")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Bias Value")
        if "mnist" in experiment_name.lower():
            ax.set_title("Perceptron Bias Evolution (MNIST)")
        else:
            ax.set_title(f"Perceptron Bias Evolution ({experiment_name})")
        ax.legend()
        self.save_and_show(fig, save_name)
        return fig

    def plot_accuracy_evolution(self, accuracy_history: np.ndarray, save_name: str = "accuracy_curve", experiment_name: str = "mnist"):
        import numpy as np
        import matplotlib.pyplot as plt
        epochs = np.arange(1, len(accuracy_history) + 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, accuracy_history, label="Training Accuracy", color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        if "mnist" in experiment_name.lower():
            ax.set_title("Perceptron Training Accuracy (MNIST)")
        else:
            ax.set_title(f"Perceptron Training Accuracy ({experiment_name})")
        ax.legend()
        self.save_and_show(fig, save_name)
        return fig

    def create_minimal_weights_gif(self, weight_history: np.ndarray, save_path: str = "outputs/plots/perceptron_weights_minimal.gif") -> str:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import numpy as np
        import os
        if weight_history.ndim != 2 or weight_history.shape[1] != 784:
            raise ValueError("weight_history must be of shape (epochs, 784)")
        n_epochs = weight_history.shape[0]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Guarantee 28x28 pixel output: 1 inch x 1 inch at 28 dpi
        fig, ax = plt.subplots(figsize=(1, 1), dpi=28)
        im = ax.imshow(weight_history[0].reshape(28, 28), cmap="coolwarm", vmin=np.min(weight_history), vmax=np.max(weight_history))
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.patch.set_alpha(0)
        def update(epoch):
            im.set_data(weight_history[epoch].reshape(28, 28))
            return [im]
        anim = FuncAnimation(fig, update, frames=n_epochs, interval=400, blit=True)
        anim.save(save_path, writer='pillow')
        plt.close(fig)
        # Post-process with PIL to ensure 28x28 pixel GIF
        from PIL import Image, ImageSequence
        with Image.open(save_path) as im:
            frames = [frame.copy().resize((28, 28), resample=Image.NEAREST) for frame in ImageSequence.Iterator(im)]
            frames[0].save(save_path, save_all=True, append_images=frames[1:], loop=0, duration=im.info.get('duration', 100), disposal=2)
        return save_path

    def plot_final_weights_distribution(self, final_weights: np.ndarray, save_path: str = "outputs/plots/perceptron_final_weights.png", experiment_name: str = "mnist") -> str:
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        if final_weights.size != 784:
            raise ValueError("final_weights must be of size 784")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(final_weights.reshape(28, 28), cmap="coolwarm")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Weight Value")
        ax.set_title(f"Final Perceptron Weights ({experiment_name.upper()})")
        ax.axis("off")
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    def _generate_performance_insights(self, accuracy: float,
                                     precision: float, recall: float) -> str:
        """Generate performance insights for the perceptron model.

        Args:
            accuracy: Model accuracy
            precision: Model precision
            recall: Model recall

        Returns:
            Formatted performance insights string
        """
        insights = f"""Perceptron Performance Analysis:

📊 Accuracy: {accuracy:.3f}
🎯 Precision: {precision:.3f}
📈 Recall: {recall:.3f}

Educational Insights:
• Linear Separability: {'✅ Achieved' if accuracy > 0.95 else '❌ Not achieved'}
• Convergence: {'✅ Converged' if accuracy > 0.95 else '❌ May not converge'}
• Model Complexity: Simple linear classifier
• Learning Capability: Binary classification only

The Perceptron is a fundamental neural network that demonstrates:
• Linear decision boundaries
• Binary classification
• Iterative learning process
• Convergence guarantees for separable data"""

        return insights