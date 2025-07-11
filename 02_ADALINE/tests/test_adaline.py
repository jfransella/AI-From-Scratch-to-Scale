"""
Unit tests for ADALINE (Adaptive Linear Neuron) implementation.

This module provides comprehensive unit tests for all ADALINE components
including model, data loader, evaluator, and visualizer.
"""

import unittest
import numpy as np
import tempfile
import os
from typing import Dict, Any

# Add src to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.config import ADALINEConfig
from src.data_loader import ADALINEDataLoader
from src.model import ADALINE, ADALINEState
from src.evaluate import ADALINEEvaluator
from src.visualize import ADALINEVisualizer


class TestADALINEConfig(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ADALINEConfig()
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Test valid config
        self.assertIsInstance(self.config, ADALINEConfig)
        
        # Test invalid learning rate
        with self.assertRaises(ValueError):
            config = ADALINEConfig(LEARNING_RATE=-0.01)
        
        # Test invalid max epochs
        with self.assertRaises(ValueError):
            config = ADALINEConfig(MAX_EPOCHS=0)
        
        # Test invalid convergence threshold
        with self.assertRaises(ValueError):
            config = ADALINEConfig(CONVERGENCE_THRESHOLD=-1e-6)
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config_dict = self.config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn('learning_rate', config_dict)
        self.assertIn('max_epochs', config_dict)
        self.assertIn('input_size', config_dict)


class TestADALINEDataLoader(unittest.TestCase):
    """Test data loading and preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ADALINEConfig()
        self.data_loader = ADALINEDataLoader(self.config)
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        X, y = self.data_loader.generate_synthetic_data(
            n_samples=100,
            n_features=2,
            problem_type='regression'
        )
        
        self.assertEqual(X.shape, (100, 2))
        self.assertEqual(y.shape, (100, 1))
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
    
    def test_data_preprocessing(self):
        """Test data preprocessing."""
        X = np.random.randn(50, 2)
        y = np.random.randn(50, 1)
        
        X_processed, y_processed = self.data_loader.preprocess_data(X, y, fit_scaler=True)
        
        # Check bias term addition
        if self.config.ADD_BIAS_TERM:
            self.assertEqual(X_processed.shape[1], X.shape[1] + 1)
        
        # Check normalization
        if self.config.NORMALIZE_FEATURES:
            self.assertNotEqual(np.array_equal(X, X_processed), True)
    
    def test_data_splitting(self):
        """Test data splitting functionality."""
        X = np.random.randn(100, 2)
        y = np.random.randn(100, 1)
        
        data_splits = self.data_loader.split_data(X, y)
        
        self.assertIn('X_train', data_splits)
        self.assertIn('y_train', data_splits)
        self.assertIn('X_val', data_splits)
        self.assertIn('y_val', data_splits)
        self.assertIn('X_test', data_splits)
        self.assertIn('y_test', data_splits)
        
        # Check that splits are non-empty
        for key, data in data_splits.items():
            self.assertGreater(data.shape[0], 0)
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data."""
        # Test empty data
        with self.assertRaises(ValueError):
            self.data_loader.preprocess_data(np.array([]), np.array([]))
        
        # Test shape mismatch
        X = np.random.randn(10, 3)  # Wrong number of features
        y = np.random.randn(10, 1)
        with self.assertRaises(ValueError):
            self.data_loader.preprocess_data(X, y)


class TestADALINEModel(unittest.TestCase):
    """Test ADALINE model implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ADALINEConfig()
        self.model = ADALINE(
            input_size=3,  # 2 features + bias
            learning_rate=0.01,
            random_seed=42,
            cfg=self.config
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.input_size, 3)
        self.assertEqual(self.model.learning_rate, 0.01)
        self.assertFalse(self.model.is_fitted)
        self.assertIsNone(self.model.weights)
        self.assertIsNone(self.model.bias)
    
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        self.model._initialize_parameters()
        
        self.assertIsNotNone(self.model.weights)
        self.assertIsNotNone(self.model.bias)
        self.assertEqual(self.model.weights.shape, (3,))
        self.assertIsInstance(self.model.bias, (int, float, np.number))
    
    def test_forward_pass(self):
        """Test forward pass computation."""
        self.model._initialize_parameters()
        
        X = np.random.randn(10, 3)
        outputs = self.model.forward(X)
        
        self.assertEqual(outputs.shape, (10,))
        self.assertIsInstance(outputs, np.ndarray)
    
    def test_loss_computation(self):
        """Test loss computation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        
        loss = self.model._compute_loss(y_true, y_pred)
        
        self.assertIsInstance(loss, (int, float, np.number))
        self.assertGreaterEqual(loss, 0)  # Loss should be non-negative
    
    def test_gradient_computation(self):
        """Test gradient computation."""
        self.model._initialize_parameters()
        
        X = np.random.randn(10, 3)
        y_true = np.random.randn(10)
        y_pred = self.model.forward(X)
        
        weight_grads, bias_grad = self.model._compute_gradients(X, y_true, y_pred)
        
        self.assertEqual(weight_grads.shape, (3,))
        self.assertIsInstance(bias_grad, (int, float, np.number))
    
    def test_model_fitting(self):
        """Test model fitting."""
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        
        training_state = self.model.fit(X, y, max_epochs=10)
        
        self.assertTrue(self.model.is_fitted)
        self.assertIsNotNone(self.model.weights)
        self.assertIsNotNone(self.model.bias)
        self.assertIsInstance(training_state, ADALINEState)
        self.assertGreater(len(training_state.training_loss), 0)
    
    def test_model_prediction(self):
        """Test model prediction."""
        # Train model first
        X_train = np.random.randn(100, 3)
        y_train = np.random.randn(100)
        self.model.fit(X_train, y_train, max_epochs=10)
        
        # Test prediction
        X_test = np.random.randn(10, 3)
        predictions = self.model.predict(X_test)
        
        self.assertEqual(predictions.shape, (10,))
        self.assertIsInstance(predictions, np.ndarray)
    
    def test_model_scoring(self):
        """Test model scoring."""
        # Train model first
        X_train = np.random.randn(100, 3)
        y_train = np.random.randn(100)
        self.model.fit(X_train, y_train, max_epochs=10)
        
        # Test scoring
        X_test = np.random.randn(20, 3)
        y_test = np.random.randn(20)
        score = self.model.score(X_test, y_test)
        
        self.assertIsInstance(score, (int, float, np.number))
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        # Train model first
        X_train = np.random.randn(100, 3)
        y_train = np.random.randn(100)
        self.model.fit(X_train, y_train, max_epochs=10)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            self.model.save_model(tmp.name)
            model_path = tmp.name
        
        try:
            # Create new model and load
            new_model = ADALINE(input_size=3, learning_rate=0.01)
            new_model.load_model(model_path)
            
            # Test that loaded model works
            X_test = np.random.randn(10, 3)
            predictions = new_model.predict(X_test)
            self.assertEqual(predictions.shape, (10,))
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        # Test invalid input size
        with self.assertRaises(ValueError):
            ADALINE(input_size=0, learning_rate=0.01)
        
        # Test invalid learning rate
        with self.assertRaises(ValueError):
            ADALINE(input_size=2, learning_rate=-0.01)
    
    def test_unfitted_model_errors(self):
        """Test errors when using unfitted model."""
        X = np.random.randn(10, 3)
        
        # Test prediction without fitting
        with self.assertRaises(ValueError):
            self.model.predict(X)
        
        # Test scoring without fitting
        y = np.random.randn(10)
        with self.assertRaises(ValueError):
            self.model.score(X, y)


class TestADALINEEvaluator(unittest.TestCase):
    """Test evaluation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ADALINEConfig()
        self.evaluator = ADALINEEvaluator(self.config)
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = self.evaluator.evaluate_predictions(y_true, y_pred)
        
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2_score', metrics)
        
        # Check that metrics are reasonable
        self.assertGreaterEqual(metrics['mse'], 0)
        self.assertGreaterEqual(metrics['mae'], 0)
    
    def test_cross_validation(self):
        """Test cross-validation functionality."""
        # Create a simple model for testing
        model = ADALINE(input_size=3, learning_rate=0.01)
        
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        
        # Train model first
        model.fit(X, y, max_epochs=10)
        
        # Test cross-validation
        cv_results = self.evaluator.cross_validate(model, X, y, cv_folds=3)
        
        self.assertIn('mean_r2_score', cv_results)
        self.assertIn('std_r2_score', cv_results)
        self.assertIn('fold_scores', cv_results)
    
    def test_residual_analysis(self):
        """Test residual analysis."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        residuals = self.evaluator.analyze_residuals(y_true, y_pred)
        
        self.assertIn('mean_residual', residuals)
        self.assertIn('std_residual', residuals)
        self.assertIn('skewness', residuals)
        self.assertIn('kurtosis', residuals)


class TestADALINEVisualizer(unittest.TestCase):
    """Test visualization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ADALINEConfig()
        self.visualizer = ADALINEVisualizer(self.config)
        
        # Create a trained model for testing
        self.model = ADALINE(input_size=3, learning_rate=0.01)
        X_train = np.random.randn(100, 3)
        y_train = np.random.randn(100)
        self.training_state = self.model.fit(X_train, y_train, max_epochs=10)
        
        # Create data splits for testing
        self.data_splits = {
            'X_train': np.random.randn(50, 2),
            'y_train': np.random.randn(50, 1),
            'X_test': np.random.randn(20, 2),
            'y_test': np.random.randn(20, 1)
        }
    
    def test_training_progress_plot(self):
        """Test training progress plotting."""
        plot_path = self.visualizer.plot_training_progress(
            self.training_state, save=True)
        
        self.assertIsInstance(plot_path, str)
        if plot_path:  # If plot was saved
            self.assertTrue(os.path.exists(plot_path))
    
    def test_weight_evolution_plot(self):
        """Test weight evolution plotting."""
        plot_path = self.visualizer.plot_weight_evolution(
            self.training_state, save=True)
        
        self.assertIsInstance(plot_path, str)
        if plot_path:  # If plot was saved
            self.assertTrue(os.path.exists(plot_path))
    
    def test_decision_boundary_plot(self):
        """Test decision boundary plotting."""
        plot_path = self.visualizer.plot_decision_boundary(
            self.model, self.data_splits, save=True)
        
        self.assertIsInstance(plot_path, str)
        if plot_path:  # If plot was saved
            self.assertTrue(os.path.exists(plot_path))


class TestADALINEIntegration(unittest.TestCase):
    """Integration tests for complete ADALINE pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ADALINEConfig(
            MAX_EPOCHS=10,  # Short training for testing
            CONVERGENCE_THRESHOLD=1e-3
        )
    
    def test_complete_pipeline(self):
        """Test complete training pipeline."""
        # Create components
        data_loader = ADALINEDataLoader(self.config)
        model = ADALINE(input_size=3, learning_rate=0.01, cfg=self.config)
        evaluator = ADALINEEvaluator(self.config)
        visualizer = ADALINEVisualizer(self.config)
        
        # Generate data
        X, y = data_loader.generate_synthetic_data(
            n_samples=100, n_features=2, problem_type='regression'
        )
        
        # Preprocess data
        X_processed, y_processed = data_loader.preprocess_data(X, y, fit_scaler=True)
        
        # Split data
        data_splits = data_loader.split_data(X_processed, y_processed)
        
        # Train model
        training_state = model.fit(
            data_splits['X_train'], 
            data_splits['y_train'].flatten(),
            data_splits['X_val'],
            data_splits['y_val'].flatten()
        )
        
        # Evaluate model
        y_pred = model.predict(data_splits['X_test'])
        metrics = evaluator.evaluate_predictions(
            data_splits['y_test'].flatten(), y_pred
        )
        
        # Test that training completed
        self.assertTrue(model.is_fitted)
        self.assertGreater(len(training_state.training_loss), 0)
        self.assertIn('mse', metrics)
        self.assertIn('r2_score', metrics)


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2) 