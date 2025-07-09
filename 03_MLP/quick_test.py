#!/usr/bin/env python3
"""Quick test to verify the refactored MLP implementation."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from config import MLPConfig
        from data_loader import DataLoader
        from model import MLP
        from train import MLPTrainer
        from evaluate import MLPEvaluator
        from visualize import MLPVisualizer
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic MLP functionality."""
    try:
        from config import MLPConfig
        from model import MLP
        import numpy as np
        
        # Create config and model
        config = MLPConfig()
        model = MLP(config)
        
        # Test forward pass
        X = np.random.randn(10, config.input_size)
        y = np.random.randint(0, config.output_size, size=(10,))
        
        predictions = model.predict(X)
        print(f"✅ Forward pass successful - predictions shape: {predictions.shape}")
        
        # Test training step
        loss = model._compute_loss(X, y)
        print(f"✅ Loss computation successful - loss: {loss:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Quick MLP Refactoring Test")
    print("=" * 40)
    
    success = True
    success &= test_imports()
    success &= test_basic_functionality()
    
    if success:
        print("\n🎉 All tests passed! Refactoring is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
