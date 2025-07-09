#!/usr/bin/env python3
"""Test script to verify all improvements work correctly."""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_improvements():
    """Test all the improvements we made."""
    print("🧪 Testing Python Best Practices Improvements...")
    
    # Test imports
    try:
        from model import Perceptron
        from evaluate import evaluate_model
        from data_loader import load_perceptron_data
        from constants import DEFAULT_LEARNING_RATE, DEFAULT_ITERATIONS
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test model creation with type hints and validation
    try:
        model = Perceptron(learning_rate=0.1, n_iters=5, random_seed=42)
        print("✅ Model creation with parameters successful")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test input validation
    try:
        # This should raise an error
        invalid_model = Perceptron(learning_rate=-0.1)
        print("❌ Input validation failed - negative learning rate should raise error")
        return False
    except ValueError:
        print("✅ Input validation working - caught negative learning rate")
    
    # Test with simple data
    try:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([0, 0, 0, 1], dtype=np.int32)
        
        model.fit(X, y)
        predictions = model.predict(X)
        metrics = evaluate_model(model, X, y)
        
        print(f"✅ Training and evaluation successful")
        print(f"   Accuracy: {metrics['accuracy']:.2f}")
        print(f"   Model repr: {repr(model)}")
    except Exception as e:
        print(f"❌ Training/evaluation failed: {e}")
        return False
    
    # Test constants usage
    try:
        model_with_defaults = Perceptron()
        assert model_with_defaults.learning_rate == DEFAULT_LEARNING_RATE
        assert model_with_defaults.n_iters == DEFAULT_ITERATIONS
        print("✅ Constants usage verified")
    except Exception as e:
        print(f"❌ Constants test failed: {e}")
        return False
    
    print("🎉 All tests passed! Python best practices implementation successful.")
    return True

if __name__ == "__main__":
    success = test_improvements()
    sys.exit(0 if success else 1)
