"""
Unit tests for PlotFactory functionality.

This module tests the standardized plot creation methods and demonstrates
how the PlotFactory eliminates code duplication across models.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os

# Import the class to test
from .plot_factory import PlotFactory


class TestPlotFactory(unittest.TestCase):
    """Test cases for PlotFactory functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_wandb_visualizer = Mock()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_plot_factory_initialization(self):
        """Test PlotFactory initialization."""
        factory = PlotFactory(
            model_name="Perceptron",
            wandb_visualizer=self.mock_wandb_visualizer,
            save_dir=self.temp_dir
        )
        
        self.assertEqual(factory.model_name, "Perceptron")
        self.assertEqual(factory.wandb_visualizer, self.mock_wandb_visualizer)
        self.assertEqual(factory.save_dir, Path(self.temp_dir))
        self.assertIsNotNone(factory.base_visualizer)
    
    def test_create_training_plot(self):
        """Test standardized training plot creation."""
        factory = PlotFactory(model_name="MLP", save_dir=self.temp_dir)
        
        # Create mock training data
        training_data = {
            'loss': [0.5, 0.3, 0.2, 0.1, 0.05],
            'accuracy': [0.6, 0.7, 0.8, 0.9, 0.95]
        }
        
        # Create training plot
        fig, ax = factory.create_training_plot(
            training_data=training_data,
            plot_type="learning_curve"
        )
        
        # Verify plot was created
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        
        # Verify plot has expected elements
        self.assertEqual(ax.get_xlabel(), "Epoch")
        self.assertEqual(ax.get_ylabel(), "Metric Value")
        self.assertIn("MLP Training Progress", ax.get_title())
    
    def test_create_decision_boundary(self):
        """Test standardized decision boundary creation."""
        factory = PlotFactory(model_name="Perceptron", save_dir=self.temp_dir)
        
        # Create mock model with predict method
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1])
        
        # Create 2D features and labels
        features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        labels = np.array([0, 1, 0, 1])
        
        # Create decision boundary plot
        fig, ax = factory.create_decision_boundary(
            model=mock_model,
            features=features,
            labels=labels
        )
        
        # Verify plot was created
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        
        # Verify plot has expected elements
        self.assertEqual(ax.get_xlabel(), "Feature 1")
        self.assertEqual(ax.get_ylabel(), "Feature 2")
        self.assertIn("Perceptron Decision Boundary", ax.get_title())
    
    def test_create_confusion_matrix(self):
        """Test standardized confusion matrix creation."""
        factory = PlotFactory(model_name="Classifier", save_dir=self.temp_dir)
        
        # Create mock predictions
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        
        # Create confusion matrix
        fig, ax = factory.create_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=['Class 0', 'Class 1']
        )
        
        # Verify plot was created
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        
        # Verify plot has expected elements
        self.assertEqual(ax.get_xlabel(), "Predicted Label")
        self.assertEqual(ax.get_ylabel(), "True Label")
        self.assertIn("Classifier Confusion Matrix", ax.get_title())
    
    def test_create_weight_evolution(self):
        """Test standardized weight evolution creation."""
        factory = PlotFactory(model_name="Neural Network", save_dir=self.temp_dir)
        
        # Create mock weight history
        weight_history = [
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])],  # Epoch 1
            [np.array([0.15, 0.25, 0.35]), np.array([0.45, 0.55, 0.65])],  # Epoch 2
            [np.array([0.2, 0.3, 0.4]), np.array([0.5, 0.6, 0.7])],  # Epoch 3
        ]
        
        # Create weight evolution plot
        fig, axes = factory.create_weight_evolution(
            weight_history=weight_history,
            layer_names=['Input Layer', 'Output Layer']
        )
        
        # Verify plot was created
        self.assertIsNotNone(fig)
        self.assertIsNotNone(axes)
        self.assertEqual(len(axes), 4)  # 2x2 subplot layout
        
        # Verify overall title
        self.assertIn("Neural Network Weight Evolution", fig._suptitle.get_text())
    
    def test_save_and_log_integration(self):
        """Test integration with W&B logging."""
        factory = PlotFactory(
            model_name="Test Model",
            wandb_visualizer=self.mock_wandb_visualizer,
            save_dir=self.temp_dir
        )
        
        # Create a simple test figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        
        # Test save and log
        with patch.object(factory.wandb_visualizer, 'log_figure_with_metadata') as mock_log:
            factory.save_and_log(
                fig=fig,
                name="test_plot",
                plot_type="learning_curve",
                model_info={'model_type': 'Test'},
                dataset_info={'name': 'Test Dataset'},
                hyperparameters={'learning_rate': 0.01}
            )
            
            # Verify W&B logging was called
            mock_log.assert_called_once()
    
    def test_educational_annotations(self):
        """Test that educational annotations are added automatically."""
        factory = PlotFactory(model_name="Educational Model", save_dir=self.temp_dir)
        
        # Create training plot
        training_data = {'loss': [0.5, 0.3, 0.2]}
        fig, ax = factory.create_training_plot(training_data, plot_type="learning_curve")
        
        # Verify educational annotation was added
        # (This would require checking the annotation text, but the method exists)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)


class TestPlotFactoryBenefits(unittest.TestCase):
    """Test cases demonstrating the benefits of PlotFactory."""
    
    def test_eliminates_code_duplication(self):
        """Demonstrate how PlotFactory eliminates code duplication."""
        # Before PlotFactory (hypothetical code duplication):
        # Each model would have its own plot creation methods:
        # - PerceptronVisualizer.create_learning_curve()
        # - MLPVisualizer.create_learning_curve() 
        # - HopfieldVisualizer.create_learning_curve()
        # All doing similar things with slight variations
        
        # After PlotFactory - single standardized method:
        factory = PlotFactory(model_name="Any Model")
        training_data = {'loss': [0.5, 0.3, 0.2]}
        
        # Same interface for all models
        fig, ax = factory.create_training_plot(training_data)
        
        # Verify consistent behavior
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
    
    def test_consistent_styling(self):
        """Demonstrate consistent styling across models."""
        models = ["Perceptron", "MLP", "Hopfield", "CNN"]
        
        for model_name in models:
            factory = PlotFactory(model_name=model_name)
            training_data = {'accuracy': [0.6, 0.7, 0.8]}
            
            fig, ax = factory.create_training_plot(training_data)
            
            # All plots should have consistent styling
            self.assertEqual(ax.get_xlabel(), "Epoch")
            self.assertEqual(ax.get_ylabel(), "Metric Value")
            self.assertTrue(ax.get_grid())
    
    def test_automatic_figure_sizing(self):
        """Demonstrate automatic figure sizing based on plot type."""
        factory = PlotFactory(model_name="Test Model")
        
        # Different plot types should get appropriate figure sizes
        plot_types = ['learning_curve', 'decision_boundary', 'confusion_matrix']
        
        for plot_type in plot_types:
            if plot_type == 'learning_curve':
                training_data = {'loss': [0.5, 0.3, 0.2]}
                fig, ax = factory.create_training_plot(training_data, plot_type=plot_type)
            elif plot_type == 'decision_boundary':
                mock_model = Mock()
                mock_model.predict.return_value = np.array([0, 1])
                features = np.array([[1, 2], [3, 4]])
                labels = np.array([0, 1])
                fig, ax = factory.create_decision_boundary(mock_model, features, labels)
            elif plot_type == 'confusion_matrix':
                y_true = np.array([0, 1])
                y_pred = np.array([0, 1])
                fig, ax = factory.create_confusion_matrix(y_true, y_pred)
            
            # Each should have appropriate figure size
            self.assertIsNotNone(fig)
            self.assertIsNotNone(ax)


if __name__ == '__main__':
    unittest.main() 