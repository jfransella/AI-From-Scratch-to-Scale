# -*- coding: utf-8 -*-
"""Simple test script to verify the refactored MLP implementation."""

import sys
import os
import numpy as np
import logging

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def test_model_basic_functionality():
    """Test basic model functionality with dummy data."""
    print("Testing basic MLP functionality...")
    
    try:
        from model import MLP
        
        # Create simple binary classification dummy data
        X_dummy = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y_dummy = np.array([[0], [1], [1], [0]], dtype=np.float32)  # XOR pattern
        
        # Test model initialization
        model = MLP(
            input_size=2,
            hidden_size=4,
            output_size=1,
            learning_rate=0.1,
            epochs=10,
            random_seed=42
        )
        print("✓ Model initialization successful")
        
        # Test model representation
        print(f"Model repr: {repr(model)}")
        
        # Test forward pass
        predictions = model.predict(X_dummy)
        print(f"✓ Forward pass successful, predictions shape: {predictions.shape}")
        
        # Test training (just a few epochs to verify no errors)
        model.fit(X_dummy, y_dummy)
        print("✓ Training completed successfully")
        
        # Test predictions after training
        final_predictions = model.predict(X_dummy)
        accuracy = model.score(X_dummy, y_dummy)
        print(f"✓ Final accuracy: {accuracy:.4f}")
        
        # Test save/load
        model.save_model("test_model.npz")
        print("✓ Model saving successful")
        
        # Create new model and load weights
        model2 = MLP(input_size=2, hidden_size=4, output_size=1)
        model2.load_model("test_model.npz")
        print("✓ Model loading successful")
        
        # Verify loaded model gives same predictions
        loaded_predictions = model2.predict(X_dummy)
        assert np.allclose(final_predictions, loaded_predictions), "Loaded model predictions don't match!"
        print("✓ Model save/load consistency verified")
        
        # Clean up
        if os.path.exists("test_model.npz"):
            os.remove("test_model.npz")
        
        print("✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_data_loader_functionality():
    """Test data loading functionality with dummy data."""
    print("\nTesting data loader functionality...")
    
    try:
        from data_loader import load_logic_gate_data
        import pandas as pd
        
        # Create a dummy CSV file for testing
        dummy_data = pd.DataFrame({
            'input1': [0, 0, 1, 1],
            'input2': [0, 1, 0, 1],
            'output': [0, 1, 1, 0]
        })
        dummy_data.to_csv("test_data.csv", index=False)
        
        # Test loading
        X, y = load_logic_gate_data("test_data.csv")
        print(f"✓ Data loading successful, X shape: {X.shape}, y shape: {y.shape}")
        
        # Verify data types
        assert X.dtype == np.float32, f"Expected X dtype float32, got {X.dtype}"
        assert y.dtype == np.float32, f"Expected y dtype float32, got {y.dtype}"
        print("✓ Data types are correct")
        
        # Clean up
        if os.path.exists("test_data.csv"):
            os.remove("test_data.csv")
        
        print("✅ Data loader tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_evaluate_functionality():
    """Test evaluation functionality."""
    print("\nTesting evaluation functionality...")
    
    try:
        from evaluate import evaluate_model
        from model import MLP
        
        # Create dummy data
        X_test = np.random.rand(100, 10).astype(np.float32)
        y_test = np.random.randint(0, 3, (100,))  # 3-class problem
        
        # Create dummy model
        model = MLP(input_size=10, hidden_size=5, output_size=3, epochs=1)
        model.fit(X_test[:10], np.eye(3)[y_test[:10]])  # Train on small subset
        
        # Test evaluation
        results = evaluate_model(model, X_test, y_test)
        
        # Verify results structure
        expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix', 'n_samples', 'n_classes']
        for key in expected_keys:
            assert key in results, f"Missing key in results: {key}"
        
        print(f"✓ Evaluation successful, accuracy: {results['accuracy']:.4f}")
        print("✅ Evaluation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running MLP Refactoring Tests")
    print("=" * 50)
    
    # Setup basic logging
    logging.basicConfig(level=logging.WARNING)  # Suppress info logs during testing
    
    # Run tests
    tests = [
        test_model_basic_functionality,
        test_data_loader_functionality,
        test_evaluate_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactoring was successful.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
