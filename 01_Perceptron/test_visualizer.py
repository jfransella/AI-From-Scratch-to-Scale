#!/usr/bin/env python3
"""
Test script for Perceptron Visualizer
=====================================

This script tests the Perceptron visualizer to ensure it can be instantiated
and used correctly with the updated shared framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the visualizer
try:
    from visualize import PerceptronVisualizer  # type: ignore
except ImportError:
    # Fallback for when running as script
    import sys
    sys.path.append('src')
    from visualize import PerceptronVisualizer  # type: ignore

def test_visualizer_initialization():
    """Test that the visualizer can be initialized correctly."""
    print("Testing PerceptronVisualizer initialization...")
    
    try:
        # Test basic initialization
        viz = PerceptronVisualizer()
        print("✓ Basic initialization successful")
        
        # Test with custom save directory
        viz = PerceptronVisualizer(save_dir="test_outputs")
        print("✓ Custom save directory initialization successful")
        
        # Test with disabled visualization
        viz = PerceptronVisualizer(enabled=False)
        print("✓ Disabled visualization initialization successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

def test_visualizer_attributes():
    """Test that the visualizer has the expected attributes."""
    print("\nTesting visualizer attributes...")
    
    try:
        viz = PerceptronVisualizer()
        
        # Check that required attributes exist
        assert hasattr(viz, 'model_name'), "model_name attribute missing"
        assert hasattr(viz, 'enabled'), "enabled attribute missing"
        assert hasattr(viz, 'confusion_matrix_viz'), "confusion_matrix_viz attribute missing"
        assert hasattr(viz, 'training_curve_viz'), "training_curve_viz attribute missing"
        assert hasattr(viz, 'decision_boundary_viz'), "decision_boundary_viz attribute missing"
        assert hasattr(viz, 'interactive_viz'), "interactive_viz attribute missing"
        assert hasattr(viz, 'advanced_viz'), "advanced_viz attribute missing"
        
        print("✓ All required attributes present")
        
        # Check model name
        assert viz.model_name == "Perceptron", f"Expected model_name 'Perceptron', got '{viz.model_name}'"
        print("✓ Model name correctly set to 'Perceptron'")
        
        return True
        
    except Exception as e:
        print(f"✗ Attribute test failed: {e}")
        return False

def test_visualizer_methods():
    """Test that the visualizer methods can be called without errors."""
    print("\nTesting visualizer methods...")
    
    try:
        viz = PerceptronVisualizer()
        
        # Create dummy data
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        errors_per_epoch = [5, 3, 2, 1, 0]
        
        # Test confusion matrix method (should return None when disabled)
        viz.enabled = False
        result = viz.plot_confusion_matrix(y_true, y_pred)
        assert result is None, "Should return None when disabled"
        print("✓ Confusion matrix method works when disabled")
        
        # Test learning curve method (should return None when disabled)
        result = viz.plot_learning_curve(errors_per_epoch)
        assert result is None, "Should return None when disabled"
        print("✓ Learning curve method works when disabled")
        
        # Re-enable for further tests
        viz.enabled = True
        
        return True
        
    except Exception as e:
        print(f"✗ Method test failed: {e}")
        return False

def test_shared_framework_integration():
    """Test that the shared framework components are properly integrated."""
    print("\nTesting shared framework integration...")
    
    try:
        viz = PerceptronVisualizer()
        
        # Test that shared framework components are initialized
        assert viz.confusion_matrix_viz is not None, "ConfusionMatrixVisualizer not initialized"
        assert viz.training_curve_viz is not None, "TrainingCurveVisualizer not initialized"
        assert viz.decision_boundary_viz is not None, "DecisionBoundaryVisualizer not initialized"
        assert viz.interactive_viz is not None, "InteractiveVisualizer not initialized"
        assert viz.advanced_viz is not None, "AdvancedVisualizer not initialized"
        
        print("✓ All shared framework components initialized")
        
        # Test that interactive and advanced visualizers have correct model names
        assert viz.interactive_viz.model_name == "Perceptron", "InteractiveVisualizer model_name incorrect"
        assert viz.advanced_viz.model_name == "Perceptron", "AdvancedVisualizer model_name incorrect"
        print("✓ Interactive and advanced visualizers have correct model names")
        
        return True
        
    except Exception as e:
        print(f"✗ Shared framework integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Perceptron Visualizer Test Suite")
    print("=" * 40)
    
    tests = [
        test_visualizer_initialization,
        test_visualizer_attributes,
        test_visualizer_methods,
        test_shared_framework_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Perceptron visualizer is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 