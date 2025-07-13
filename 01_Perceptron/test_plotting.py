#!/usr/bin/env python3
"""
Test plotting functionality of Perceptron Visualizer
==================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from visualize import PerceptronVisualizer  # type: ignore
except ImportError:
    import sys
    sys.path.append('src')
    from visualize import PerceptronVisualizer  # type: ignore

def test_plotting_functionality():
    """Test that the visualizer can create actual plots."""
    print("Testing PerceptronVisualizer plotting functionality...")
    
    try:
        # Initialize visualizer
        viz = PerceptronVisualizer(save_dir="test_outputs")
        
        # Create dummy data
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        errors_per_epoch = [8, 6, 4, 2, 1, 0, 0, 0]
        
        # Test confusion matrix plotting
        print("Testing confusion matrix plotting...")
        fig1 = viz.plot_confusion_matrix(y_true, y_pred)
        assert fig1 is not None, "Confusion matrix should return a figure"
        print("✓ Confusion matrix plotting successful")
        
        # Test learning curve plotting
        print("Testing learning curve plotting...")
        fig2 = viz.plot_learning_curve(errors_per_epoch)
        assert fig2 is not None, "Learning curve should return a figure"
        print("✓ Learning curve plotting successful")
        
        # Test with disabled visualization
        viz.enabled = False
        fig3 = viz.plot_confusion_matrix(y_true, y_pred)
        assert fig3 is None, "Should return None when disabled"
        print("✓ Disabled visualization works correctly")
        
        print("🎉 All plotting tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Plotting test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_plotting_functionality()
    sys.exit(0 if success else 1) 