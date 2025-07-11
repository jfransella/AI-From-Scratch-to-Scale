"""Test the new MLP visualizer implementation."""
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Import the new visualizer
from visualize_new import MLPVisualizer, plot_confusion_matrix

def test_visualizer():
    """Test basic functionality of the new MLP visualizer."""
    print("Testing MLPVisualizer with shared framework...")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)  # 2D for decision boundary
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple linear separation
    y_pred = np.random.choice([0, 1], size=100)
    
    # Create mock model
    class MockModel:
        def __init__(self):
            self.W1 = np.random.randn(784, 128)  # MNIST-like weights
            self.losses = [1.5, 1.2, 0.9, 0.7, 0.5]
            self.accuracies = [0.3, 0.5, 0.7, 0.8, 0.85]
        
        def predict(self, X):
            # Simple mock prediction
            return (X[:, 0] + X[:, 1] > 0).astype(int)
    
    model = MockModel()
    
    # Test visualizer initialization
    viz = MLPVisualizer()
    print("✅ MLPVisualizer initialized successfully")
    
    # Test confusion matrix
    try:
        fig, ax = viz.plot_confusion_matrix(y, y_pred, class_names=['Class 0', 'Class 1'], show=False)
        print("✅ Confusion matrix generated successfully")
        viz.cleanup_figures()
    except Exception as e:
        print(f"❌ Confusion matrix failed: {e}")
    
    # Test training curves
    try:
        fig, ax = viz.plot_training_curves(model.losses, model.accuracies, show=False)
        print("✅ Training curves generated successfully")
        viz.cleanup_figures()
    except Exception as e:
        print(f"❌ Training curves failed: {e}")
    
    # Test decision boundary
    try:
        fig, ax = viz.plot_decision_boundary(X, y, model, class_names=['Class 0', 'Class 1'], show=False)
        print("✅ Decision boundary generated successfully")
        viz.cleanup_figures()
    except Exception as e:
        print(f"❌ Decision boundary failed: {e}")
    
    # Test neuron weights
    try:
        fig, ax = viz.plot_neuron_weights(model.W1, num_neurons_to_show=4, show=False)
        print("✅ Neuron weights visualization generated successfully")
        viz.cleanup_figures()
    except Exception as e:
        print(f"❌ Neuron weights failed: {e}")
    
    # Test wrapper function
    try:
        fig = plot_confusion_matrix(y, y_pred, class_names=['Class 0', 'Class 1'], show=False)
        print("✅ Wrapper function works successfully")
    except Exception as e:
        print(f"❌ Wrapper function failed: {e}")
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    test_visualizer()
