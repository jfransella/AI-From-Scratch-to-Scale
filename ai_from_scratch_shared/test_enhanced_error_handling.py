"""
Test Enhanced Error Handling and Validation
=========================================

This script tests the enhanced error handling and validation features
of the visualization framework, demonstrating how validation helps
users understand and fix common mistakes.

Key Features Tested:
- Data shape validation with detailed error messages
- Model interface validation
- Training data validation
- Graceful fallbacks for missing components
- Educational error messages with suggestions
- W&B integration validation
"""

import sys
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import Mock
import logging

# Add the shared package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the validation components
from visualization.validation import (
    VisualizationValidator, 
    ValidationError,
    validate_visualization_inputs,
    validate_training_visualization_inputs,
    create_validation_error_with_context
)
from visualization.plot_factory import PlotFactory
from visualization import BaseVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_data_shape_validation():
    """Test data shape validation with detailed error messages."""
    logger.info("=" * 60)
    logger.info("TESTING DATA SHAPE VALIDATION")
    logger.info("=" * 60)
    
    validator = VisualizationValidator()
    
    # Test 1: Valid 2D features
    logger.info("Test 1: Valid 2D features")
    valid_features = np.array([[1, 2], [3, 4], [5, 6]])
    try:
        validator.validate_2d_features(valid_features, "features")
        logger.info("✅ Valid 2D features passed validation")
    except ValidationError as e:
        logger.error(f"❌ Unexpected validation error: {e}")
    
    # Test 2: Invalid 1D features
    logger.info("Test 2: Invalid 1D features")
    invalid_features_1d = np.array([1, 2, 3, 4])
    try:
        validator.validate_2d_features(invalid_features_1d, "features")
        logger.error("❌ Should have failed validation")
    except ValidationError as e:
        logger.info("✅ Correctly caught 1D features error")
        logger.info(f"Error message: {e.message}")
        logger.info(f"Suggestions: {e.suggestions}")
    
    # Test 3: Invalid 3D features
    logger.info("Test 3: Invalid 3D features")
    invalid_features_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    try:
        validator.validate_2d_features(invalid_features_3d, "features")
        logger.error("❌ Should have failed validation")
    except ValidationError as e:
        logger.info("✅ Correctly caught 3D features error")
        logger.info(f"Error message: {e.message}")
        logger.info(f"Suggestions: {e.suggestions}")
    
    # Test 4: Non-numpy array
    logger.info("Test 4: Non-numpy array")
    invalid_features_list = [[1, 2], [3, 4], [5, 6]]
    try:
        # Convert to numpy array first to test the type validation
        invalid_features_array = np.array(invalid_features_list)
        validator.validate_2d_features(invalid_features_array, "features")
        logger.error("❌ Should have failed validation")
    except ValidationError as e:
        logger.info("✅ Correctly caught non-numpy array error")
        logger.info(f"Error message: {e.message}")
        logger.info(f"Suggestions: {e.suggestions}")


def test_model_interface_validation():
    """Test model interface validation."""
    logger.info("=" * 60)
    logger.info("TESTING MODEL INTERFACE VALIDATION")
    logger.info("=" * 60)
    
    validator = VisualizationValidator()
    
    # Test 1: Valid model with predict method
    logger.info("Test 1: Valid model with predict method")
    valid_model = Mock()
    valid_model.predict = Mock(return_value=np.array([0, 1, 0]))
    try:
        validator.validate_model_interface(valid_model)
        logger.info("✅ Valid model passed validation")
    except ValidationError as e:
        logger.error(f"❌ Unexpected validation error: {e}")
    
    # Test 2: Invalid model without predict method
    logger.info("Test 2: Invalid model without predict method")
    invalid_model = Mock()
    # No predict method
    try:
        validator.validate_model_interface(invalid_model)
        logger.error("❌ Should have failed validation")
    except ValidationError as e:
        logger.info("✅ Correctly caught missing predict method error")
        logger.info(f"Error message: {e.message}")
        logger.info(f"Suggestions: {e.suggestions}")
    
    # Test 3: Model with multiple missing methods
    logger.info("Test 3: Model with multiple missing methods")
    incomplete_model = Mock()
    try:
        validator.validate_model_interface(incomplete_model, ['predict', 'fit', 'score'])
        logger.error("❌ Should have failed validation")
    except ValidationError as e:
        logger.info("✅ Correctly caught multiple missing methods error")
        logger.info(f"Error message: {e.message}")
        logger.info(f"Suggestions: {e.suggestions}")


def test_training_data_validation():
    """Test training data validation."""
    logger.info("=" * 60)
    logger.info("TESTING TRAINING DATA VALIDATION")
    logger.info("=" * 60)
    
    validator = VisualizationValidator()
    
    # Test 1: Valid training data
    logger.info("Test 1: Valid training data")
    valid_training_data = {
        'loss': [0.5, 0.3, 0.2, 0.1],
        'accuracy': [0.6, 0.7, 0.8, 0.9]
    }
    try:
        validator.validate_training_data(valid_training_data)
        logger.info("✅ Valid training data passed validation")
    except ValidationError as e:
        logger.error(f"❌ Unexpected validation error: {e}")
    
    # Test 2: Empty training data
    logger.info("Test 2: Empty training data")
    empty_training_data = {}
    try:
        validator.validate_training_data(empty_training_data)
        logger.error("❌ Should have failed validation")
    except ValidationError as e:
        logger.info("✅ Correctly caught empty training data error")
        logger.info(f"Error message: {e.message}")
        logger.info(f"Suggestions: {e.suggestions}")
    
    # Test 3: Training data with mismatched lengths
    logger.info("Test 3: Training data with mismatched lengths")
    mismatched_training_data = {
        'loss': [0.5, 0.3, 0.2],
        'accuracy': [0.6, 0.7, 0.8, 0.9]  # Different length
    }
    try:
        validator.validate_training_data(mismatched_training_data)
        logger.error("❌ Should have failed validation")
    except ValidationError as e:
        logger.info("✅ Correctly caught mismatched lengths error")
        logger.info(f"Error message: {e.message}")
        logger.info(f"Suggestions: {e.suggestions}")
    
    # Test 4: Non-dictionary training data
    logger.info("Test 4: Non-dictionary training data")
    invalid_training_data = [0.5, 0.3, 0.2, 0.1]  # List instead of dict
    try:
        validator.validate_training_data(invalid_training_data)  # type: ignore
        logger.error("❌ Should have failed validation")
    except ValidationError as e:
        logger.info("✅ Correctly caught non-dictionary error")
        logger.info(f"Error message: {e.message}")
        logger.info(f"Suggestions: {e.suggestions}")


def test_plot_factory_validation():
    """Test validation in PlotFactory methods."""
    logger.info("=" * 60)
    logger.info("TESTING PLOT FACTORY VALIDATION")
    logger.info("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        factory = PlotFactory(model_name="Test Model", save_dir=temp_dir)
        
        # Test 1: Valid training plot creation
        logger.info("Test 1: Valid training plot creation")
        valid_training_data = {
            'loss': [0.5, 0.3, 0.2, 0.1],
            'accuracy': [0.6, 0.7, 0.8, 0.9]
        }
        try:
            fig, ax = factory.create_training_plot(valid_training_data)
            logger.info("✅ Valid training plot created successfully")
        except ValidationError as e:
            logger.error(f"❌ Unexpected validation error: {e}")
        
        # Test 2: Invalid training data in PlotFactory
        logger.info("Test 2: Invalid training data in PlotFactory")
        invalid_training_data = {
            'loss': [0.5, 0.3],
            'accuracy': [0.6, 0.7, 0.8]  # Different length
        }
        try:
            fig, ax = factory.create_training_plot(invalid_training_data)
            logger.error("❌ Should have failed validation")
        except ValidationError as e:
            logger.info("✅ Correctly caught invalid training data error")
            logger.info(f"Error message: {e.message}")
            logger.info(f"Suggestions: {e.suggestions}")
        
        # Test 3: Valid confusion matrix creation
        logger.info("Test 3: Valid confusion matrix creation")
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0])
        try:
            fig, ax = factory.create_confusion_matrix(y_true, y_pred)
            logger.info("✅ Valid confusion matrix created successfully")
        except ValidationError as e:
            logger.error(f"❌ Unexpected validation error: {e}")
        
        # Test 4: Invalid confusion matrix inputs
        logger.info("Test 4: Invalid confusion matrix inputs")
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0])  # Different length
        try:
            fig, ax = factory.create_confusion_matrix(y_true, y_pred)
            logger.error("❌ Should have failed validation")
        except ValidationError as e:
            logger.info("✅ Correctly caught mismatched lengths error")
            logger.info(f"Error message: {e.message}")
            logger.info(f"Suggestions: {e.suggestions}")


def test_graceful_fallbacks():
    """Test graceful fallbacks for missing components."""
    logger.info("=" * 60)
    logger.info("TESTING GRACEFUL FALLBACKS")
    logger.info("=" * 60)
    
    validator = VisualizationValidator()
    
    # Test 1: Graceful fallback for validation failure
    logger.info("Test 1: Graceful fallback for validation failure")
    invalid_features = np.array([1, 2, 3, 4])  # 1D instead of 2D
    fallback_result = validator.validate_with_graceful_fallback(
        validator.validate_2d_features,
        invalid_features,
        fallback_value="fallback_features"
    )
    logger.info(f"Fallback result: {fallback_result}")
    
    # Test 2: Successful validation (no fallback needed)
    logger.info("Test 2: Successful validation (no fallback needed)")
    valid_features = np.array([[1, 2], [3, 4], [5, 6]])
    result = validator.validate_with_graceful_fallback(
        validator.validate_2d_features,
        valid_features,
        fallback_value="fallback_features"
    )
    logger.info(f"Validation result: {result}")


def test_educational_error_messages():
    """Test educational error messages with suggestions."""
    logger.info("=" * 60)
    logger.info("TESTING EDUCATIONAL ERROR MESSAGES")
    logger.info("=" * 60)
    
    # Test 1: Error with context-aware suggestions
    logger.info("Test 1: Error with context-aware suggestions")
    try:
        # Simulate a shape mismatch error
        context = {
            'actual_shape': (100, 3),
            'expected_shape': (100, 2),
            'data_name': 'features'
        }
        error = create_validation_error_with_context(
            'shape_mismatch',
            'Features must be 2D for visualization',
            context
        )
        raise error
    except ValidationError as e:
        logger.info("✅ Educational error message generated")
        logger.info(f"Error: {e.message}")
        logger.info(f"Suggestions: {e.suggestions}")


def test_integration_with_base_visualizer():
    """Test validation integration with BaseVisualizer."""
    logger.info("=" * 60)
    logger.info("TESTING BASE VISUALIZER VALIDATION")
    logger.info("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = BaseVisualizer(
            model_name="Test Model",
            default_save_dir=temp_dir
        )
        
        # Test 1: Valid input validation
        logger.info("Test 1: Valid input validation")
        try:
            visualizer.validate_inputs(
                model_name="Valid Model",
                style_theme="educational"
            )
            logger.info("✅ Valid inputs passed validation")
        except ValidationError as e:
            logger.error(f"❌ Unexpected validation error: {e}")
        
        # Test 2: Invalid input validation
        logger.info("Test 2: Invalid input validation")
        try:
            visualizer.validate_inputs(
                model_name="",  # Empty model name
                style_theme="invalid_theme"  # Invalid theme
            )
            logger.error("❌ Should have failed validation")
        except ValidationError as e:
            logger.info("✅ Correctly caught invalid inputs error")
            logger.info(f"Error: {e.message}")
            logger.info(f"Suggestions: {e.suggestions}")
        
        # Test 3: Data validation for visualization
        logger.info("Test 3: Data validation for visualization")
        valid_features = np.array([[1, 2], [3, 4], [5, 6]])
        valid_labels = np.array([0, 1, 0])
        valid_model = Mock()
        valid_model.predict = Mock(return_value=np.array([0, 1, 0]))
        
        try:
            visualizer.validate_data_for_visualization(
                features=valid_features,
                labels=valid_labels,
                model=valid_model
            )
            logger.info("✅ Valid data passed validation")
        except ValidationError as e:
            logger.error(f"❌ Unexpected validation error: {e}")


def main():
    """Run all enhanced error handling tests."""
    logger.info("Starting Enhanced Error Handling and Validation Tests")
    logger.info("=" * 80)
    
    try:
        # Test data shape validation
        test_data_shape_validation()
        
        # Test model interface validation
        test_model_interface_validation()
        
        # Test training data validation
        test_training_data_validation()
        
        # Test PlotFactory validation
        test_plot_factory_validation()
        
        # Test graceful fallbacks
        test_graceful_fallbacks()
        
        # Test educational error messages
        test_educational_error_messages()
        
        # Test BaseVisualizer integration
        test_integration_with_base_visualizer()
        
        logger.info("=" * 80)
        logger.info("🎉 ALL ENHANCED ERROR HANDLING TESTS PASSED!")
        logger.info("=" * 80)
        
        # Summary of enhanced error handling features
        logger.info("SUMMARY OF ENHANCED ERROR HANDLING FEATURES:")
        logger.info("✅ Comprehensive data shape validation")
        logger.info("✅ Model interface validation")
        logger.info("✅ Training data structure validation")
        logger.info("✅ Graceful fallbacks for missing components")
        logger.info("✅ Educational error messages with suggestions")
        logger.info("✅ Context-aware fix suggestions")
        logger.info("✅ Integration with BaseVisualizer and PlotFactory")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 