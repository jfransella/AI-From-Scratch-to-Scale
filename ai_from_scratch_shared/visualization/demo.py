"""
Demonstration of the Shared Visualization Framework
=================================================

This script demonstrates the usage of the shared visualization components
and validates that the framework is working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the parent directory to path to import our shared module
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ai_from_scratch_shared.visualization import (
        BaseVisualizer,
        ConfusionMatrixVisualizer,
        TrainingCurveVisualizer,
        DecisionBoundaryVisualizer,
        DataDistributionVisualizer,
        apply_educational_theme,
        add_mathematical_context
    )
    print("✓ Successfully imported shared visualization components")
except ImportError as e:
    print(f"✗ Failed to import shared visualization components: {e}")
    sys.exit(1)

def demonstrate_base_visualizer():
    """Demonstrate BaseVisualizer functionality."""
    print("\n--- BaseVisualizer Demonstration ---")
    
    # Create a base visualizer
    viz = BaseVisualizer(model_name="Demo", default_save_dir=Path("demo_outputs"))
    
    # Create a simple figure
    fig, ax = viz.create_figure(figsize='default')
    
    # Add some sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, color=viz.colors['primary'], linewidth=2, label='sin(x)')
    
    # Apply styling
    viz.apply_consistent_styling(ax, "Sine Wave Demo", "x", "y")
    ax.legend()
    
    # Add educational annotation
    viz.add_educational_annotation(
        ax,
        "This demonstrates the BaseVisualizer class\nwith consistent educational styling.",
        position="top_right"
    )
    
    # Add mathematical context
    add_mathematical_context(
        ax,
        concept="Sine Function",
        formula=r"y = \sin(x)",
        explanation="The sine function oscillates between -1 and 1 with period 2π."
    )
    
    # Save and show
    save_path = viz.save_and_show(fig, "base_visualizer_demo.png", show=False)
    print(f"✓ Created demo plot: {save_path}")
    
    # Clean up
    viz.cleanup_figures()

def demonstrate_training_curves():
    """Demonstrate TrainingCurveVisualizer."""
    print("\n--- TrainingCurveVisualizer Demonstration ---")
    
    # Create sample training data
    epochs = 50
    train_losses = [1.0 * np.exp(-i/10) + 0.1 * np.random.random() for i in range(epochs)]
    val_losses = [1.2 * np.exp(-i/12) + 0.15 * np.random.random() for i in range(epochs)]
    
    # Create visualizer
    viz = TrainingCurveVisualizer()
    
    # Plot training curves
    fig, ax = viz.plot_loss_curve(
        train_losses, 
        val_losses, 
        title="Demo Training Curves",
        show=False
    )
    
    print("✓ Created training curves demonstration")
    viz.cleanup_figures()

def demonstrate_data_distribution():
    """Demonstrate DataDistributionVisualizer."""
    print("\n--- DataDistributionVisualizer Demonstration ---")
    
    # Generate sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Two classes with different distributions
    class_0 = np.random.normal([0, 0], [1, 1], (n_samples//2, 2))
    class_1 = np.random.normal([2, 2], [1.5, 0.8], (n_samples//2, 2))
    
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Create visualizer
    viz = DataDistributionVisualizer()
    
    # Plot feature distributions
    fig, axes = viz.plot_feature_distributions(
        X, y,
        feature_names=["Feature 1", "Feature 2"],
        class_names=["Class A", "Class B"],
        show=False
    )
    
    # Plot class balance
    fig2, ax2 = viz.plot_class_balance(
        y,
        class_names=["Class A", "Class B"],
        show=False
    )
    
    print("✓ Created data distribution demonstrations")
    viz.cleanup_figures()

def main():
    """Run all demonstrations."""
    print("Shared Visualization Framework Demonstration")
    print("=" * 50)
    
    # Apply educational theme globally
    apply_educational_theme()
    print("✓ Applied educational theme")
    
    # Create output directory
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Created output directory: {output_dir}")
    
    # Run demonstrations
    try:
        demonstrate_base_visualizer()
        demonstrate_training_curves()
        demonstrate_data_distribution()
        
        print("\n" + "=" * 50)
        print("✅ All demonstrations completed successfully!")
        print(f"Check the '{output_dir}' directory for generated plots.")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
