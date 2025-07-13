"""
Phase 2 Test Suite: Advanced Visualization Features
==================================================

This module provides comprehensive testing for Phase 2 features including
interactive visualizations and advanced plot types.

Test Coverage:
- InteractiveVisualizer functionality
- AdvancedVisualizer functionality
- Real-time plot updates
- Interactive decision boundaries
- Advanced plot types (gradient flow, attention, feature importance)
- Educational enhancements
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import tempfile
import time
from pathlib import Path
import logging

from .interactive import InteractiveVisualizer, InteractionType, InteractiveElement
from .advanced import AdvancedVisualizer, PlotType, LayerInfo
from .validation import ValidationError

logger = logging.getLogger(__name__)


class TestInteractiveVisualizer(unittest.TestCase):
    """Test cases for InteractiveVisualizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = InteractiveVisualizer(
            model_name="InteractiveTest",
            default_save_dir=self.temp_dir
        )
        
        # Create test data
        np.random.seed(42)
        self.X = np.random.randn(50, 2)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)
        
        # Create mock model
        class MockModel:
            def predict(self, X):
                return (X[:, 0] + X[:, 1] > 0).astype(int)
        
        self.model = MockModel()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.visualizer.cleanup_interactive_elements()
    
    def test_initialization(self):
        """Test InteractiveVisualizer initialization."""
        self.assertEqual(self.visualizer.model_name, "InteractiveTest")
        self.assertTrue(self.visualizer.enable_animations)
        self.assertEqual(len(self.visualizer.interactive_elements), 0)
        self.assertEqual(len(self.visualizer.animation_objects), 0)
    
    def test_create_interactive_decision_boundary(self):
        """Test interactive decision boundary creation."""
        fig, ax = self.visualizer.create_interactive_decision_boundary(
            model=self.model,
            features=self.X,
            labels=self.y
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        self.assertIsNotNone(self.visualizer.current_figure)
        
        # Check that interactive elements were created
        self.assertGreater(len(self.visualizer.interactive_elements), 0)
        
        # Check element types
        element_types = [elem.element_type for elem in self.visualizer.interactive_elements]
        self.assertIn(InteractionType.SLIDER, element_types)
        self.assertIn(InteractionType.BUTTON, element_types)
    
    def test_create_real_time_training_plot(self):
        """Test real-time training plot creation."""
        # Mock update callback
        def update_callback():
            return {
                'loss': [1.0 * np.exp(-i/10) + 0.1 * np.random.random() for i in range(10)],
                'accuracy': [0.5 + 0.4 * (1 - np.exp(-i/8)) + 0.05 * np.random.random() for i in range(10)]
            }
        
        fig, ax = self.visualizer.create_real_time_training_plot(update_callback)
        
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        self.assertIsNotNone(self.visualizer.current_figure)
        
        # Check that animation was created
        self.assertGreater(len(self.visualizer.animation_objects), 0)
    
    def test_add_parameter_slider(self):
        """Test adding parameter slider."""
        fig, ax = self.visualizer.create_figure()
        # Handle both single Axes and numpy array of axes
        if isinstance(ax, np.ndarray):
            ax = ax.flat[0]  # Use first subplot
        
        def slider_callback(val):
            pass
        
        slider = self.visualizer.add_parameter_slider(
            ax=ax,
            param_name="learning_rate",
            min_val=0.001,
            max_val=0.1,
            init_val=0.01,
            callback=slider_callback,
            description="Learning rate control"
        )
        
        self.assertIsNotNone(slider)
        self.assertEqual(len(self.visualizer.interactive_elements), 1)
        self.assertEqual(self.visualizer.interactive_elements[0].element_type, InteractionType.SLIDER)
    
    def test_add_hover_tooltip(self):
        """Test adding hover tooltips."""
        fig, ax = self.visualizer.create_figure()
        # Handle both single Axes and numpy array of axes
        if isinstance(ax, np.ndarray):
            ax = ax.flat[0]  # Use first subplot
        
        data_points = np.random.randn(10, 2)
        tooltip_data = [f"Point {i}" for i in range(10)]
        
        # This should not raise an error
        self.visualizer.add_hover_tooltip(ax, data_points, tooltip_data)
    
    def test_export_interactive_plot(self):
        """Test interactive plot export."""
        fig, ax = self.visualizer.create_interactive_decision_boundary(
            model=self.model,
            features=self.X,
            labels=self.y
        )
        
        # Test PNG export
        result = self.visualizer.export_interactive_plot("test_interactive.png", format="png")
        self.assertIsInstance(result, Path)
        
        # Test that file exists
        self.assertTrue(result.exists())
    
    def test_cleanup_interactive_elements(self):
        """Test cleanup of interactive elements."""
        # Create some interactive elements
        fig, ax = self.visualizer.create_interactive_decision_boundary(
            model=self.model,
            features=self.X,
            labels=self.y
        )
        
        # Verify elements exist
        self.assertGreater(len(self.visualizer.interactive_elements), 0)
        
        # Cleanup
        self.visualizer.cleanup_interactive_elements()
        
        # Verify cleanup
        self.assertEqual(len(self.visualizer.interactive_elements), 0)
        self.assertEqual(len(self.visualizer.animation_objects), 0)
    
    def test_get_interactive_info(self):
        """Test getting interactive information."""
        info = self.visualizer.get_interactive_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn("num_elements", info)
        self.assertIn("element_types", info)
        self.assertIn("num_animations", info)
        self.assertIn("current_figure", info)


class TestAdvancedVisualizer(unittest.TestCase):
    """Test cases for AdvancedVisualizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = AdvancedVisualizer(
            model_name="AdvancedTest",
            default_save_dir=self.temp_dir
        )
        
        # Create test data
        np.random.seed(42)
        self.gradients = [np.random.randn(10, 10) for _ in range(5)]
        self.layer_names = ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Output"]
        
        self.feature_names = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"]
        self.importance_scores = np.random.rand(5)
        
        self.epochs = list(range(1, 21))
        self.learning_rates = [0.1 * (0.9 ** i) for i in range(20)]
        
        self.attention_weights = np.random.rand(8, 8)
        self.input_tokens = [f"Token {i}" for i in range(8)]
        self.output_tokens = [f"Token {i}" for i in range(8)]
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test AdvancedVisualizer initialization."""
        self.assertEqual(self.visualizer.model_name, "AdvancedTest")
    
    def test_create_gradient_flow(self):
        """Test gradient flow visualization."""
        fig, ax = self.visualizer.create_gradient_flow(
            gradients=self.gradients,
            layer_names=self.layer_names
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        
        # Check that plot elements exist
        self.assertGreater(len(ax.patches), 0)  # Should have bars
        self.assertGreater(len(ax.texts), 0)     # Should have annotations
    
    def test_create_attention_heatmap(self):
        """Test attention heatmap visualization."""
        fig, ax = self.visualizer.create_attention_heatmap(
            attention_weights=self.attention_weights,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        
        # Check that heatmap was created
        self.assertGreater(len(ax.images), 0)
    
    def test_create_feature_importance(self):
        """Test feature importance visualization."""
        fig, ax = self.visualizer.create_feature_importance(
            feature_names=self.feature_names,
            importance_scores=self.importance_scores
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        
        # Check that bars were created
        self.assertGreater(len(ax.patches), 0)
    
    def test_create_learning_rate_schedule(self):
        """Test learning rate schedule visualization."""
        fig, ax = self.visualizer.create_learning_rate_schedule(
            epochs=self.epochs,
            learning_rates=self.learning_rates
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        
        # Check that line was created
        self.assertGreater(len(ax.lines), 0)
    
    def test_create_model_comparison(self):
        """Test model comparison visualization."""
        model_names = ["Model A", "Model B", "Model C"]
        metrics = {
            "accuracy": [0.85, 0.92, 0.88],
            "precision": [0.82, 0.89, 0.85],
            "recall": [0.87, 0.94, 0.90]
        }
        
        fig, ax = self.visualizer.create_model_comparison(
            model_names=model_names,
            metrics=metrics
        )
        
        self.assertIsInstance(fig, Figure)
        # Handle both single Axes and list of Axes
        if isinstance(ax, list):
            for single_ax in ax:
                self.assertIsInstance(single_ax, Axes)
        else:
            self.assertIsInstance(ax, Axes)
        
        # Check that bars were created (for bar chart)
        if isinstance(ax, list):
            # For radar charts, check first subplot
            self.assertGreater(len(ax[0].patches), 0)
        else:
            # For bar charts
            self.assertGreater(len(ax.patches), 0)
    
    def test_create_network_architecture(self):
        """Test network architecture visualization."""
        layers = [
            LayerInfo("Input", (784,), (128,), "ReLU", 100480),
            LayerInfo("Hidden 1", (128,), (64,), "ReLU", 8256),
            LayerInfo("Hidden 2", (64,), (32,), "ReLU", 2080),
            LayerInfo("Output", (32,), (10,), "Softmax", 330)
        ]
        
        fig, ax = self.visualizer.create_network_architecture(layers=layers)
        
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        
        # Check that rectangles were created
        self.assertGreater(len(ax.patches), 0)
    
    def test_validation_errors(self):
        """Test validation error handling."""
        # Test empty gradients
        with self.assertRaises(ValidationError):
            self.visualizer.create_gradient_flow([])
        
        # Test invalid attention weights
        with self.assertRaises(ValidationError):
            self.visualizer.create_attention_heatmap(np.random.rand(3, 4))  # Non-square
        
        # Test mismatched feature names and scores
        with self.assertRaises(ValidationError):
            self.visualizer.create_feature_importance(
                feature_names=["A", "B"],
                importance_scores=np.random.rand(3)
            )
    
    def test_plot_type_enum(self):
        """Test PlotType enum."""
        self.assertEqual(PlotType.GRADIENT_FLOW.value, "gradient_flow")
        self.assertEqual(PlotType.ATTENTION_HEATMAP.value, "attention_heatmap")
        self.assertEqual(PlotType.FEATURE_IMPORTANCE.value, "feature_importance")
        self.assertEqual(PlotType.LEARNING_RATE_SCHEDULE.value, "learning_rate_schedule")
        self.assertEqual(PlotType.MODEL_COMPARISON.value, "model_comparison")
        self.assertEqual(PlotType.NETWORK_ARCHITECTURE.value, "network_architecture")
        self.assertEqual(PlotType.LOSS_LANDSCAPE.value, "loss_landscape")
    
    def test_layer_info_dataclass(self):
        """Test LayerInfo dataclass."""
        layer = LayerInfo(
            name="Test Layer",
            input_shape=(10,),
            output_shape=(5,),
            activation="ReLU",
            parameters=55
        )
        
        self.assertEqual(layer.name, "Test Layer")
        self.assertEqual(layer.input_shape, (10,))
        self.assertEqual(layer.output_shape, (5,))
        self.assertEqual(layer.activation, "ReLU")
        self.assertEqual(layer.parameters, 55)


class TestPhase2Integration(unittest.TestCase):
    """Integration tests for Phase 2 features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.interactive_visualizer = InteractiveVisualizer("IntegrationTest")
        self.advanced_visualizer = AdvancedVisualizer("IntegrationTest")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.interactive_visualizer.cleanup_interactive_elements()
    
    def test_phase2_workflow(self):
        """Test complete Phase 2 workflow."""
        # Create interactive decision boundary
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        class MockModel:
            def predict(self, X):
                return (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = MockModel()
        
        # Interactive visualization
        fig1, ax1 = self.interactive_visualizer.create_interactive_decision_boundary(
            model=model, features=X, labels=y
        )
        
        # Advanced visualization
        gradients = [np.random.randn(10, 10) for _ in range(3)]
        fig2, ax2 = self.advanced_visualizer.create_gradient_flow(gradients)
        
        # Feature importance
        feature_names = ["Feature A", "Feature B", "Feature C"]
        importance_scores = np.random.rand(3)
        fig3, ax3 = self.advanced_visualizer.create_feature_importance(
            feature_names, importance_scores
        )
        
        # Verify all figures were created
        self.assertIsInstance(fig1, Figure)
        self.assertIsInstance(fig2, Figure)
        self.assertIsInstance(fig3, Figure)
        
        # Verify interactive elements
        self.assertGreater(len(self.interactive_visualizer.interactive_elements), 0)
        
        # Cleanup
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)


def run_phase2_tests():
    """Run all Phase 2 tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestInteractiveVisualizer,
        TestAdvancedVisualizer,
        TestPhase2Integration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("RUNNING PHASE 2 ADVANCED VISUALIZATION FEATURES TESTS")
    print("=" * 60)
    
    # Run all tests
    success = run_phase2_tests()
    
    print("=" * 60)
    if success:
        print("✅ ALL PHASE 2 TESTS PASSED!")
        print("🎉 Advanced visualization features are working correctly!")
    else:
        print("❌ SOME PHASE 2 TESTS FAILED!")
        print("🔧 Please check the test output above for details.")
    print("=" * 60) 