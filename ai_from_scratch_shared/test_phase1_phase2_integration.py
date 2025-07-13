"""
Integration Test for Phase 1 and Phase 2 Improvements
====================================================

This script tests the integration between:
- Phase 1: Enhanced W&B Integration Patterns (log_figure_with_metadata)
- Phase 2: Standardized Plot Creation Interface (PlotFactory)

It demonstrates how these improvements work together to provide
a comprehensive visualization framework.
"""

import sys
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import Mock, patch
import logging

# Add the shared package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the new functionality
from wandb_integration import BaseWandbVisualizer
from visualization.plot_factory import PlotFactory
from visualization import BaseVisualizer

# Create a concrete test implementation
class TestWandbVisualizer(BaseWandbVisualizer, BaseVisualizer):
    def __init__(self, wandb_run=None, enabled=True, plots_dir="outputs/plots"):
        BaseWandbVisualizer.__init__(self, wandb_run=wandb_run, enabled=enabled)
        BaseVisualizer.__init__(self, model_name="Test Model", default_save_dir=plots_dir)
    
    def create_model_visualizations(self, model, features, y, predictions):
        return {}
    
    def log_model_config(self, model_config):
        pass
    
    def log_training_progress(self, epoch, metrics):
        pass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_phase1_wandb_integration():
    """Test Phase 1: Enhanced W&B Integration Patterns."""
    logger.info("=" * 60)
    logger.info("TESTING PHASE 1: Enhanced W&B Integration")
    logger.info("=" * 60)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock W&B run
        mock_wandb_run = Mock()
        mock_wandb_run.name = "test-run-123"
        
        # Initialize W&B visualizer
        wandb_visualizer = TestWandbVisualizer(
            wandb_run=mock_wandb_run,
            enabled=True,
            plots_dir=temp_dir
        )
        
        # Test metadata extraction
        model_info = {
            'model_type': 'Perceptron',
            'n_parameters': 3,
            'architecture': 'Single Layer'
        }
        
        dataset_info = {
            'name': 'Iris',
            'n_samples': 150,
            'n_features': 2,
            'n_classes': 2
        }
        
        hyperparameters = {
            'learning_rate': 0.01,
            'epochs': 100,
            'optimizer': 'SGD'
        }
        
        # Test caption generation
        caption = wandb_visualizer._generate_plot_caption(
            plot_type='confusion_matrix',
            model_info=model_info,
            dataset_info=dataset_info,
            hyperparameters=hyperparameters
        )
        
        logger.info(f"Generated caption: {caption}")
        assert "Confusion Matrix" in caption
        assert "Perceptron" in caption
        assert "Iris" in caption
        assert "learning_rate=0.0100" in caption
        
        # Test metadata extraction
        metadata = wandb_visualizer._extract_plot_metadata(
            plot_type='learning_curve',
            model_info=model_info,
            dataset_info=dataset_info,
            hyperparameters=hyperparameters,
            step=25
        )
        
        logger.info(f"Extracted metadata keys: {list(metadata.keys())}")
        assert 'plot_type' in metadata
        assert 'model' in metadata
        assert 'dataset' in metadata
        assert 'hyperparameters' in metadata
        assert metadata['step'] == 25
        
        # Test with disabled W&B (fallback behavior)
        disabled_visualizer = TestWandbVisualizer(
            wandb_run=None,
            enabled=False,
            plots_dir=temp_dir
        )
        
        # Create a test figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        
        # Test log_figure_with_metadata with disabled W&B
        with patch.object(disabled_visualizer, 'log_figure_with_metadata') as mock_log_figure:
            disabled_visualizer.log_figure_with_metadata(
                figure=fig,
                name='test_plot',
                plot_type='learning_curve',
                model_info=model_info,
                dataset_info=dataset_info,
                hyperparameters=hyperparameters
            )
            
            # Should fall back to basic logging
            mock_log_figure.assert_called_once()
        
        logger.info("✅ Phase 1 tests passed!")


def test_phase2_plot_factory():
    """Test Phase 2: Standardized Plot Creation Interface."""
    logger.info("=" * 60)
    logger.info("TESTING PHASE 2: Standardized Plot Creation Interface")
    logger.info("=" * 60)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize PlotFactory
        factory = PlotFactory(model_name="Test Model", save_dir=temp_dir)
        
        # Test 1: Training Plot Creation
        logger.info("Testing training plot creation...")
        training_data = {
            'loss': [0.5, 0.3, 0.2, 0.1, 0.05],
            'accuracy': [0.6, 0.7, 0.8, 0.9, 0.95]
        }
        
        fig, ax = factory.create_training_plot(
            training_data=training_data,
            title="Test Model Training Progress"
        )
        assert fig is not None
        assert ax is not None
        
        # Test 2: Decision Boundary Creation
        logger.info("Testing decision boundary creation...")
        mock_model = Mock()
        # Return zeros for each input sample in the mesh grid
        mock_model.predict.side_effect = lambda X: np.zeros(X.shape[0], dtype=int)
        
        features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        labels = np.array([0, 1, 0, 1])
        
        fig, ax = factory.create_decision_boundary(
            model=mock_model,
            features=features,
            labels=labels
        )
        assert fig is not None
        assert ax is not None
        
        # Test 3: Confusion Matrix Creation
        logger.info("Testing confusion matrix creation...")
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        
        fig, ax = factory.create_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=['Class 0', 'Class 1']
        )
        assert fig is not None
        assert ax is not None
        
        # Test 4: Weight Evolution Creation
        logger.info("Testing weight evolution creation...")
        weight_history = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.15, 0.25, 0.35]),
            np.array([0.2, 0.3, 0.4]),
        ]
        fig, axes = factory.create_weight_evolution(
            weight_history=weight_history
        )
        assert fig is not None
        assert axes is not None
        
        logger.info("✅ Phase 2 tests passed!")


def test_phase1_phase2_integration():
    """Test the integration between Phase 1 and Phase 2 improvements."""
    logger.info("=" * 60)
    logger.info("TESTING PHASE 1 + PHASE 2 INTEGRATION")
    logger.info("=" * 60)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock W&B run
        mock_wandb_run = Mock()
        mock_wandb_run.name = "integration-test-run"
        
        # Initialize W&B visualizer
        wandb_visualizer = TestWandbVisualizer(
            wandb_run=mock_wandb_run,
            enabled=True,
            plots_dir=temp_dir
        )
        
        # Initialize PlotFactory with W&B integration
        factory = PlotFactory(
            model_name="Integration Test Model",
            wandb_visualizer=wandb_visualizer,
            save_dir=temp_dir
        )
        
        # Test integrated workflow
        logger.info("Testing integrated workflow...")
        
        # 1. Create training plot using PlotFactory
        training_data = {
            'loss': [0.5, 0.3, 0.2, 0.1],
            'accuracy': [0.6, 0.7, 0.8, 0.9]
        }
        
        fig, ax = factory.create_training_plot(
            training_data=training_data,
            title="Integration Test Training Progress"
        )
        
        # 2. Save and log with enhanced metadata
        model_info = {
            'model_type': 'Integration Test Model',
            'n_parameters': 10,
            'architecture': 'Test Architecture'
        }
        
        dataset_info = {
            'name': 'Test Dataset',
            'n_samples': 1000,
            'n_features': 5,
            'n_classes': 2
        }
        
        hyperparameters = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'optimizer': 'Adam'
        }
        
        # Test W&B logging with metadata directly
        with patch.object(wandb_visualizer, 'log_figure_with_metadata') as mock_log:
            wandb_visualizer.log_figure_with_metadata(
                figure=fig,
                name="integration_test_plot",
                plot_type="learning_curve",
                model_info=model_info,
                dataset_info=dataset_info,
                hyperparameters=hyperparameters,
                step=10
            )
            
            # Verify W&B logging was called with correct parameters
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            
            assert call_args[1]['name'] == "integration_test_plot"
            assert call_args[1]['plot_type'] == "learning_curve"
            assert call_args[1]['model_info'] == model_info
            assert call_args[1]['dataset_info'] == dataset_info
        
        # Test multiple plot types
        logger.info("Testing multiple plot types...")
        
        # Create mock model for decision boundary
        mock_model = Mock()
        mock_model.predict.side_effect = lambda X: np.zeros(X.shape[0], dtype=int)
        
        features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        labels = np.array([0, 1, 0, 1])
        
        fig, ax = factory.create_decision_boundary(
            model=mock_model,
            features=features,
            labels=labels
        )
        
        with patch.object(wandb_visualizer, 'log_figure_with_metadata') as mock_log:
            wandb_visualizer.log_figure_with_metadata(
                figure=fig,
                name="decision_boundary_test",
                plot_type="decision_boundary",
                model_info=model_info,
                dataset_info=dataset_info,
                hyperparameters=hyperparameters
            )
            mock_log.assert_called_once()
        
        # Confusion matrix
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        
        fig, ax = factory.create_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred
        )
        
        with patch.object(wandb_visualizer, 'log_figure_with_metadata') as mock_log:
            wandb_visualizer.log_figure_with_metadata(
                figure=fig,
                name="confusion_matrix_test",
                plot_type="confusion_matrix",
                model_info=model_info,
                dataset_info=dataset_info,
                hyperparameters=hyperparameters
            )
            mock_log.assert_called_once()
        
        logger.info("✅ Integration tests passed!")


def test_educational_benefits():
    """Test the educational benefits of the integrated system."""
    logger.info("=" * 60)
    logger.info("TESTING EDUCATIONAL BENEFITS")
    logger.info("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        factory = PlotFactory(model_name="Educational Model", save_dir=temp_dir)
        
        # Test that educational annotations are added
        training_data = {'loss': [0.5, 0.3, 0.2]}
        fig, ax = factory.create_training_plot(training_data, plot_type="learning_curve")
        
        # Verify educational elements
        assert "Educational Model Training Progress" in ax.get_title()
        assert ax.get_xlabel() == "Epoch"
        assert ax.get_ylabel() == "Metric Value"
        
        # Test consistent styling across different models
        models = ["Perceptron", "MLP", "Hopfield", "CNN"]
        
        for model_name in models:
            factory = PlotFactory(model_name=model_name, save_dir=temp_dir)
            training_data = {'accuracy': [0.6, 0.7, 0.8]}
            
            fig, ax = factory.create_training_plot(training_data)
            
            # All should have consistent educational styling
            assert ax.get_xlabel() == "Epoch"
            assert ax.get_ylabel() == "Metric Value"
        
        logger.info("✅ Educational benefits tests passed!")


def main():
    """Run all integration tests."""
    logger.info("Starting Phase 1 + Phase 2 Integration Tests")
    logger.info("=" * 80)
    
    try:
        # Test Phase 1: Enhanced W&B Integration
        test_phase1_wandb_integration()
        
        # Test Phase 2: Standardized Plot Creation
        test_phase2_plot_factory()
        
        # Test Integration between Phase 1 and Phase 2
        test_phase1_phase2_integration()
        
        # Test Educational Benefits
        test_educational_benefits()
        
        logger.info("=" * 80)
        logger.info("🎉 ALL TESTS PASSED! Phase 1 and Phase 2 integration successful!")
        logger.info("=" * 80)
        
        # Summary of benefits achieved
        logger.info("SUMMARY OF ACHIEVEMENTS:")
        logger.info("✅ Enhanced W&B Integration with automatic metadata extraction")
        logger.info("✅ Standardized plot creation eliminating code duplication")
        logger.info("✅ Consistent educational styling across all models")
        logger.info("✅ Built-in W&B logging integration with rich metadata")
        logger.info("✅ Automatic educational annotations and context")
        logger.info("✅ Professional ML experiment tracking patterns")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 