"""
Perceptron Visualizer using Shared Framework
===========================================

This module provides Perceptron specific visualizations using the shared
visualization framework. It demonstrates how to extend the BaseVisualizer for
binary classification tasks while leveraging common components.

Key Features:
- Confusion matrix with binary classification insights
- Learning curve tracking convergence behavior
- Decision boundary visualization for 2D data
- Educational annotations about linear separability
- Professional styling consistent across models

Educational Focus:
- Linear separability concepts
- Perceptron convergence theorem
- Binary classification fundamentals
- Decision boundary interpretation
- Historical context of neural networks
"""

from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.figure import Figure

# Import shared visualization framework
from ai_from_scratch_shared.visualization import (
    BaseVisualizer
)

# Handle both relative and absolute imports for constants
try:
    from .constants import DECISION_BOUNDARY_RESOLUTION
except ImportError:
    from constants import DECISION_BOUNDARY_RESOLUTION

logger = logging.getLogger(__name__)


class PerceptronVisualizer(BaseVisualizer):
    """Perceptron-specific visualizer extending the shared framework.

    Provides comprehensive visualization capabilities for Perceptron models,
    focusing on binary classification, linear separability, and convergence
    behavior with educational context and professional styling.
    """

    def __init__(self, save_dir: Union[str, Path] = "outputs/plots", enabled: bool = True):
        """Initialize the Perceptron visualizer.

        Args:
            save_dir: Directory to save visualization files
            enabled: Whether to enable visualization generation
        """
        super().__init__(model_name="Perceptron", default_save_dir=save_dir)
        self.enabled = enabled

        # Perceptron-specific color scheme
        self.perceptron_colors = {
            'class_0': '#FF6B6B',  # Warm red for class 0
            'class_1': '#4ECDC4',  # Teal for class 1
            'decision_boundary': '#2C3E50',  # Dark blue-gray for boundary
            'misclassified': '#FFD93D',  # Warning yellow for errors
            'convergence': '#6C5CE7'  # Purple for learning curves
        }

        logger.info("PerceptronVisualizer initialized (enabled: %s)", enabled)

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             title: str = "Perceptron Confusion Matrix",
                             save_name: Optional[str] = "confusion_matrix") -> Optional[Figure]:
        """Create enhanced confusion matrix with binary classification insights.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names for the classes
            title: Plot title
            save_name: Filename for saving

        Returns:
            matplotlib Figure object or None if disabled
        """
        if not self.enabled:
            return None
        logger.info("📊 Generating enhanced confusion matrix...")
        if class_names is None:
            class_names = ['Class 0', 'Class 1']
        # Create figure with shared framework styling
        fig, _ = self.create_figure(figsize=(10, 8))
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Create main confusion matrix subplot
        ax_main = plt.subplot(2, 2, (1, 3))
        # Enhanced heatmap with custom styling
        im = ax_main.imshow(cm, interpolation='nearest', cmap='Blues', alpha=0.8)
        ax_main.set_title(title, fontweight='bold', pad=20, fontsize=14)
        # Add text annotations with percentages
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percentage = (count / total) * 100
                ax_main.text(j, i, f'{count}\n({percentage:.1f}%)',
                           ha="center", va="center",
                           color="white" if cm[i, j] > cm.max() / 2 else "black",
                           fontweight='bold', fontsize=12)
        # Styling
        ax_main.set_xlabel('Predicted Label', fontweight='bold')
        ax_main.set_ylabel('True Label', fontweight='bold')
        ax_main.set_xticks(range(len(class_names)))
        ax_main.set_yticks(range(len(class_names)))
        ax_main.set_xticklabels(class_names)
        ax_main.set_yticklabels(class_names)
        # Add colorbar
        plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
        # Add binary classification metrics in sidebar
        ax_metrics = plt.subplot(2, 2, 2)
        ax_metrics.axis('off')
        # Calculate binary metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics_text = f"""Binary Classification Metrics:

Accuracy: {accuracy:.3f}
Precision: {precision:.3f}
Recall (Sensitivity): {recall:.3f}
Specificity: {specificity:.3f}

Confusion Matrix Elements:
True Positives: {tp}
False Positives: {fp}
True Negatives: {tn}
False Negatives: {fn}"""
        ax_metrics.text(
            0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5",
            facecolor=self.colors['background_light'],
            alpha=0.8), fontsize=10
        )
        # Add educational context
        ax_context = plt.subplot(2, 2, 4)
        ax_context.axis('off')
        context_text = """📚 Educational Context:

• Confusion Matrix: Visualizes classification
  performance for binary problems

• Perceptron: Linear classifier that learns
  optimal decision boundary for linearly
  separable data

• Perfect Separation: Perceptron guarantees
  convergence for linearly separable data

• Misclassifications: May indicate non-linear
  separability or insufficient training"""

        ax_context.text(
            0.05, 0.95, context_text,
            transform=ax_context.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5",
            facecolor=self.colors['accent_light'],
            alpha=0.7)
        )
        # Note: add_performance_insights expects an Axes object, not Figure
        # We'll skip this for now as it requires restructuring the plot
        plt.tight_layout()
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
        epochs = np.arange(1, len(errors_per_epoch) + 1)

        # Create figure with shared framework styling
        fig, _ = self.create_figure(figsize=(12, 8))

        # Main learning curve
        ax_main = plt.subplot(2, 2, (1, 2))
        # Plot the learning curve with gradient effect
        ax_main.plot(epochs, errors_per_epoch,
                    color=self.perceptron_colors['convergence'],
                    linewidth=2.5, marker='o', markersize=4,
                    label='Misclassifications', alpha=0.8)

        # Add convergence line if converged
        if len(errors_per_epoch) > 0 and errors_per_epoch[-1] == 0:
            convergence_epoch = len(errors_per_epoch)
            for i in range(len(errors_per_epoch) - 1, -1, -1):
                if errors_per_epoch[i] > 0:
                    convergence_epoch = i + 2
                    break

            ax_main.axvline(
                x=convergence_epoch, color=self.colors['success'],
                linestyle='--', alpha=0.7, linewidth=2,
                label=f'Convergence (Epoch {convergence_epoch})'
            )

        # Styling
        ax_main.set_title(title, fontweight='bold', pad=20, fontsize=14)
        ax_main.set_xlabel('Epoch', fontweight='bold')
        ax_main.set_ylabel('Number of Misclassifications', fontweight='bold')
        ax_main.grid(True, alpha=0.3)
        ax_main.legend()

        # Add mathematical context
        ax_math = plt.subplot(2, 2, 3)
        ax_math.axis('off')

        math_text = """🔢 Mathematical Context:

Perceptron Learning Rule:
w(t+1) = w(t) + η(y - ŷ)x

Where:
• w: weight vector
• η: learning rate
• y: true label
• ŷ: predicted label
• x: input vector

Convergence Theorem:
For linearly separable data, the perceptron
is guaranteed to converge in finite steps."""

        ax_math.text(0.05, 0.95, math_text, transform=ax_math.transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['background_light'],
                             alpha=0.8))

        # Add convergence analysis
        ax_analysis = plt.subplot(2, 2, 4)
        ax_analysis.axis('off')

        total_epochs = len(errors_per_epoch)
        final_errors = errors_per_epoch[-1] if errors_per_epoch else 0
        max_errors = max(errors_per_epoch) if errors_per_epoch else 0

        analysis_text = f"""📊 Convergence Analysis:

Total Epochs: {total_epochs}
Final Errors: {final_errors}
Maximum Errors: {max_errors}
Converged: {'✅ Yes' if final_errors == 0 else '❌ No'}

Learning Behavior:
{'• Successful convergence' if final_errors == 0 else '• May need more epochs'}
{'• Data appears linearly separable' if final_errors == 0 else '• Check data separability'}"""

        ax_analysis.text(0.05, 0.95, analysis_text, transform=ax_analysis.transAxes,
                        verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['accent_light'],
                                 alpha=0.7))

        plt.tight_layout()

        if save_name:
            self.save_and_show(fig, save_name)

        return fig

    def plot_decision_boundary(self, model, features: np.ndarray, y: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             title: str = "Perceptron Decision Boundary",
                             save_name: Optional[str] = "decision_boundary") -> Optional[Figure]:
        """Create enhanced decision boundary visualization for 2D data.

        Args:
            model: Trained perceptron model with predict method
            features: Input features (must be 2D)
            y: True labels
            class_names: Names for the classes
            title: Plot title
            save_name: Filename for saving

        Returns:
            matplotlib Figure object or None if disabled or not 2D
        """
        if not self.enabled:
            return None

        if features.shape[1] != 2:
            logger.warning("Decision boundary plot only supported for 2D data")
            return None

        logger.info("🎯 Generating enhanced decision boundary...")

        if class_names is None:
            class_names = ['Class 0', 'Class 1']

        # Create figure
        fig, _ = self.create_figure(figsize=(12, 8))

        # Main decision boundary plot
        ax_main = plt.subplot(1, 2, 1)

        # Create mesh for decision boundary
        h = DECISION_BOUNDARY_RESOLUTION
        x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
        y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))

        # Make predictions on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        predictions = model.predict(mesh_points)
        predictions = predictions.reshape(xx.shape)

        # Plot decision boundary with custom colors
        colors_boundary = [self.perceptron_colors['class_0'],
                         self.perceptron_colors['class_1']]
        ax_main.contourf(xx, yy, predictions, alpha=0.4, colors=colors_boundary, levels=1)

        # Plot data points with enhanced styling
        unique_labels = np.unique(y)
        for i, label in enumerate(unique_labels):
            mask = y == label
            ax_main.scatter(features[mask, 0], features[mask, 1],
                          c=colors_boundary[i], label=class_names[i],
                          s=100, alpha=0.8, edgecolors='black', linewidth=2)

        # Draw decision boundary line
        try:
            # Calculate decision boundary line for linear classifier
            weights = model.weights if hasattr(model, 'weights') else None
            bias = model.bias if hasattr(model, 'bias') else 0

            if weights is not None and len(weights) >= 2:
                # Decision boundary: w0*x0 + w1*x1 + bias = 0
                # Solving for x1: x1 = -(w0*x0 + bias) / w1
                if abs(weights[1]) > 1e-10:  # Avoid division by zero
                    x_boundary = np.linspace(x_min, x_max, 100)
                    y_boundary = -(weights[0] * x_boundary + bias) / weights[1]
                    ax_main.plot(x_boundary, y_boundary, 'k-', linewidth=3,
                               alpha=0.8, label='Decision Boundary')
        except (ValueError, ZeroDivisionError, AttributeError) as e:
            logger.debug("Could not draw analytical decision boundary: %s", e)

        # Styling
        ax_main.set_xlim(xx.min(), xx.max())
        ax_main.set_ylim(yy.min(), yy.max())
        ax_main.set_xlabel('Feature 1', fontweight='bold')
        ax_main.set_ylabel('Feature 2', fontweight='bold')
        ax_main.set_title(title, fontweight='bold', pad=20)
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)

        # Add educational context panel
        ax_context = plt.subplot(1, 2, 2)
        ax_context.axis('off')

        context_text = """📚 Decision Boundary Analysis:

🎯 Linear Separation:
The perceptron learns a linear decision boundary
that separates the two classes. This boundary
is defined by: w₀x₀ + w₁x₁ + b = 0

🧠 Learning Process:
• Initialize weights randomly
• For each misclassified point:
  - Adjust weights toward correct classification
• Repeat until convergence

✅ Convergence Guarantee:
If data is linearly separable, the perceptron
will find a separating hyperplane in finite steps.

📊 Visual Interpretation:
• Points are colored by true class
• Background shows predicted regions
• Black line shows learned boundary
• Misclassified points indicate complexity

🔍 Key Insights:
• Simple yet powerful algorithm
• Foundation of neural networks
• Limited to linear problems
• Historically significant (1957)"""

        ax_context.text(0.05, 0.95, context_text,
                       transform=ax_context.transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox={
                           "boxstyle": "round,pad=0.8",
                           "facecolor": self.colors['background_light'],
                           "alpha": 0.9
                       }
        )

        plt.tight_layout()

        if save_name:
            self.save_and_show(fig, save_name)

        return fig

    def generate_all_visualizations(self, model, features: np.ndarray, y: np.ndarray,
                                  y_pred: np.ndarray, errors_per_epoch: List[int],
                                  class_names: Optional[List[str]] = None) -> Dict[str, Figure]:
        """Generate all standard Perceptron visualizations.

        Args:
            model: Trained perceptron model
            features: Input features
            y: True labels
            y_pred: Predicted labels
            errors_per_epoch: Training error history
            class_names: Names for classes

        Returns:
            Dictionary mapping visualization names to Figure objects
        """
        if not self.enabled:
            return {}

        logger.info("============================================================")
        logger.info("GENERATING EDUCATIONAL VISUALIZATIONS (Shared Framework)")
        logger.info("============================================================")

        figures = {}

        # Generate all visualizations
        figures['confusion_matrix'] = self.plot_confusion_matrix(
            y, y_pred, class_names
        )

        figures['learning_curve'] = self.plot_learning_curve(
            errors_per_epoch, save_name="learning_curve"
        )

        # Only generate decision boundary for 2D data
        if features.shape[1] == 2:
            figures['decision_boundary'] = self.plot_decision_boundary(
                model, features, y, class_names
            )

        # Summary
        logger.info("============================================================")
        logger.info("VISUALIZATION SUMMARY (Shared Framework)")
        logger.info("============================================================")
        generated_count = sum(1 for fig in figures.values() if fig is not None)
        logger.info("✅ Successfully generated %d visualizations:", generated_count)
        for name, fig in figures.items():
            if fig is not None:
                logger.info("   • %s", name.replace('_', ' ').title())

        logger.info("📚 Enhanced with shared framework features:")
        logger.info("   • Mathematical context annotations")
        logger.info("   • Performance insights")
        logger.info("   • Educational explanations")
        logger.info("   • Consistent styling across models")
        logger.info("============================================================")

        return figures

    def _generate_performance_insights(self, accuracy: float,
                                     precision: float, recall: float) -> str:
        """Generate performance insights for the model."""
        # Determine overall performance level
        if accuracy >= 0.95:
            level = "Excellent ⭐⭐⭐"
            overall_message = ("Perfect or near-perfect classification! "
                             "The perceptron has successfully learned the decision boundary.")
        elif accuracy >= 0.85:
            level = "Good ⭐⭐"
            overall_message = ("Strong performance with room for improvement. "
                             "Consider feature engineering or more training.")
        elif accuracy >= 0.70:
            level = "Fair ⭐"
            overall_message = ("Moderate performance. Data may not be linearly separable "
                             "or needs preprocessing.")
        else:
            level = "Poor"
            overall_message = ("Low performance suggests data is not linearly separable. "
                             "Consider non-linear approaches.")

        # Add metric-specific insights
        metric_insights = []

        if precision < 0.8:
            metric_insights.append("Low precision suggests many false positives - "
                                 "model is too eager to predict positive class")
        elif precision > 0.95:
            metric_insights.append("High precision - model rarely makes false positive errors")

        if recall < 0.8:
            metric_insights.append("Low recall suggests many false negatives - "
                                 "model misses positive cases")
        elif recall > 0.95:
            metric_insights.append("High recall - model rarely misses positive cases")

        # Balance analysis
        if abs(precision - recall) > 0.2:
            if precision > recall:
                metric_insights.append("Precision-focused: Model is conservative, "
                                    "prioritizing accuracy over coverage")
            else:
                metric_insights.append("Recall-focused: Model is aggressive, "
                                    "prioritizing coverage over accuracy")
        else:
            metric_insights.append("Balanced performance between precision and recall")

        # Combine insights
        insights = f"Performance Level: {level}\n{overall_message}"
        if metric_insights:
            insights += ("\n\nDetailed Analysis:\n" +
                       "\n".join(f"• {insight}" for insight in metric_insights))

        return insights


# Note: Legacy wrapper functions and Visualizer class have been removed.
# Use PerceptronVisualizer directly for all visualization needs.
