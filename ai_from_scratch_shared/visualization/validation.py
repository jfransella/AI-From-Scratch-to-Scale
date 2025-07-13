"""
Validation Module for Visualization Framework
==========================================

This module provides comprehensive validation for the visualization framework,
including data shape validation, model state validation, graceful fallbacks,
and detailed error messages with actionable suggestions.

Key Features:
- Data shape and type validation
- Model state and interface validation
- Graceful fallbacks for missing components
- Detailed error messages with suggestions
- Validation for W&B integration
- Educational error messages for learning

Educational Focus:
- Help users understand common mistakes
- Provide actionable suggestions for fixes
- Explain validation requirements clearly
- Guide users toward best practices
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import logging
from pathlib import Path
import warnings
import os

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors with detailed messages."""
    
    def __init__(self, message: str, suggestions: Optional[List[str]] = None):
        self.message = message
        self.suggestions = suggestions or []
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format the error message with suggestions."""
        msg = f"Validation Error: {self.message}"
        if self.suggestions:
            msg += "\n\nSuggestions:\n"
            for i, suggestion in enumerate(self.suggestions, 1):
                msg += f"  {i}. {suggestion}\n"
        return msg


class VisualizationValidator:
    """
    Comprehensive validator for visualization framework components.
    
    This class provides validation for:
    - Data shapes and types
    - Model interfaces and states
    - W&B integration requirements
    - File paths and permissions
    - Configuration parameters
    """
    
    @staticmethod
    def validate_data_shape(data: np.ndarray, 
                          expected_shape: Tuple[int, ...],
                          data_name: str = "data") -> None:
        """
        Validate data shape with detailed error messages.
        
        Args:
            data: Data array to validate
            expected_shape: Expected shape tuple
            data_name: Name of the data for error messages
            
        Raises:
            ValidationError: If shape doesn't match expected
        """
        if not isinstance(data, np.ndarray):
            raise ValidationError(
                f"{data_name} must be a numpy array, got {type(data)}",
                [
                    f"Convert to numpy array: np.array({data_name})",
                    f"Use np.asarray({data_name}) for automatic conversion"
                ]
            )
        
        if data.shape != expected_shape:
            raise ValidationError(
                f"{data_name} shape {data.shape} doesn't match expected {expected_shape}",
                [
                    f"Reshape data: {data_name}.reshape{expected_shape}",
                    f"Check data loading: ensure {data_name} has correct dimensions",
                    f"Verify data preprocessing steps"
                ]
            )
    
    @staticmethod
    def validate_2d_features(features: np.ndarray, 
                           data_name: str = "features") -> None:
        """
        Validate that features are 2D for visualization.
        
        Args:
            features: Feature array to validate
            data_name: Name of the features for error messages
            
        Raises:
            ValidationError: If features are not 2D
        """
        if not isinstance(features, np.ndarray):
            raise ValidationError(
                f"{data_name} must be a numpy array, got {type(features)}",
                [
                    f"Convert to numpy array: np.array({data_name})",
                    f"Use np.asarray({data_name}) for automatic conversion"
                ]
            )
        
        if features.ndim != 2:
            raise ValidationError(
                f"{data_name} must be 2D for visualization, got {features.ndim}D",
                [
                    f"Reshape to 2D: {data_name}.reshape(-1, 2) for 2 features",
                    f"Select 2 features: {data_name}[:, [0, 1]]",
                    f"Use PCA to reduce to 2D: from sklearn.decomposition import PCA"
                ]
            )
    
    @staticmethod
    def validate_model_interface(model: Any, 
                               required_methods: List[str] = None) -> None:
        """
        Validate that model has required interface methods.
        
        Args:
            model: Model object to validate
            required_methods: List of required method names
            
        Raises:
            ValidationError: If model lacks required methods
        """
        if required_methods is None:
            required_methods = ['predict']
        
        missing_methods = []
        for method in required_methods:
            if not hasattr(model, method):
                missing_methods.append(method)
        
        if missing_methods:
            raise ValidationError(
                f"Model missing required methods: {missing_methods}",
                [
                    f"Implement {method}() method in your model class",
                    f"Ensure model is properly trained before visualization",
                    f"Check model initialization and training steps"
                ]
            )
    
    @staticmethod
    def validate_training_data(training_data: Dict[str, List[float]],
                             required_metrics: List[str] = None) -> None:
        """
        Validate training data structure and content.
        
        Args:
            training_data: Training data dictionary
            required_metrics: List of required metric names
            
        Raises:
            ValidationError: If training data is invalid
        """
        if not isinstance(training_data, dict):
            raise ValidationError(
                f"training_data must be a dictionary, got {type(training_data)}",
                [
                    "Use format: {'loss': [...], 'accuracy': [...]}",
                    "Ensure training_data is a dict with metric names as keys"
                ]
            )
        
        if not training_data:
            raise ValidationError(
                "training_data is empty",
                [
                    "Provide training metrics in dictionary format",
                    "Check that training loop is logging metrics",
                    "Verify training data collection"
                ]
            )
        
        # Check that all metric lists have same length
        lengths = [len(values) for values in training_data.values()]
        if len(set(lengths)) > 1:
            raise ValidationError(
                f"All metric lists must have same length, got lengths: {lengths}",
                [
                    "Ensure all metrics are logged for same number of epochs",
                    "Check training loop for missing metric logging",
                    "Verify metric collection consistency"
                ]
            )
        
        # Validate required metrics if specified
        if required_metrics:
            missing_metrics = [metric for metric in required_metrics 
                             if metric not in training_data]
            if missing_metrics:
                raise ValidationError(
                    f"Missing required metrics: {missing_metrics}",
                    [
                        f"Add {metric} to training loop logging",
                        "Check metric collection in training code",
                        "Verify metric names match exactly"
                    ]
                )
    
    @staticmethod
    def validate_labels(labels: np.ndarray, 
                       expected_classes: Optional[List] = None,
                       data_name: str = "labels") -> None:
        """
        Validate label array structure and content.
        
        Args:
            labels: Label array to validate
            expected_classes: Expected class values
            data_name: Name of the labels for error messages
            
        Raises:
            ValidationError: If labels are invalid
        """
        if not isinstance(labels, np.ndarray):
            raise ValidationError(
                f"{data_name} must be a numpy array, got {type(labels)}",
                [
                    f"Convert to numpy array: np.array({data_name})",
                    f"Use np.asarray({data_name}) for automatic conversion"
                ]
            )
        
        if labels.ndim != 1:
            raise ValidationError(
                f"{data_name} must be 1D, got {labels.ndim}D",
                [
                    f"Flatten labels: {data_name}.flatten()",
                    f"Reshape to 1D: {data_name}.reshape(-1)",
                    "Check data loading and preprocessing"
                ]
            )
        
        if expected_classes is not None:
            unique_labels = np.unique(labels)
            unexpected = [label for label in unique_labels 
                         if label not in expected_classes]
            if unexpected:
                raise ValidationError(
                    f"Unexpected label values: {unexpected}",
                    [
                        f"Map labels to expected classes: {expected_classes}",
                        "Check data preprocessing and label encoding",
                        "Verify class mapping is correct"
                    ]
                )
    
    @staticmethod
    def validate_file_path(path: Union[str, Path], 
                          must_exist: bool = False,
                          must_be_writable: bool = False) -> None:
        """
        Validate file path structure and permissions.
        
        Args:
            path: File path to validate
            must_exist: Whether file must exist
            must_be_writable: Whether directory must be writable
            
        Raises:
            ValidationError: If path is invalid
        """
        path_obj = Path(path)
        
        if must_exist and not path_obj.exists():
            raise ValidationError(
                f"File does not exist: {path}",
                [
                    "Check file path spelling and location",
                    "Verify file was created successfully",
                    "Ensure file path is absolute or relative to current directory"
                ]
            )
        
        if must_be_writable:
            parent_dir = path_obj.parent
            if not parent_dir.exists():
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    raise ValidationError(
                        f"Cannot create directory: {parent_dir}",
                        [
                            "Check directory permissions",
                            "Use different save directory",
                            "Run with appropriate permissions"
                        ]
                    )
            elif not os.access(parent_dir, os.W_OK):
                raise ValidationError(
                    f"Directory not writable: {parent_dir}",
                    [
                        "Check directory permissions",
                        "Use different save directory",
                        "Run with appropriate permissions"
                    ]
                )
    
    @staticmethod
    def validate_wandb_integration(wandb_visualizer: Any,
                                  required_methods: List[str] = None) -> None:
        """
        Validate W&B integration requirements.
        
        Args:
            wandb_visualizer: W&B visualizer object
            required_methods: List of required method names
            
        Raises:
            ValidationError: If W&B integration is invalid
        """
        if required_methods is None:
            required_methods = ['log_figure_with_metadata']
        
        if wandb_visualizer is not None:
            missing_methods = []
            for method in required_methods:
                if not hasattr(wandb_visualizer, method):
                    missing_methods.append(method)
            
            if missing_methods:
                raise ValidationError(
                    f"W&B visualizer missing required methods: {missing_methods}",
                    [
                        f"Implement {method}() in W&B visualizer",
                        "Extend BaseWandbVisualizer for proper integration",
                        "Check W&B visualizer implementation"
                    ]
                )
    
    @staticmethod
    def validate_plot_parameters(**kwargs) -> None:
        """
        Validate common plot parameters.
        
        Args:
            **kwargs: Plot parameters to validate
            
        Raises:
            ValidationError: If parameters are invalid
        """
        # Validate title
        if 'title' in kwargs and kwargs['title'] is not None:
            title = kwargs['title']
            if not isinstance(title, str):
                raise ValidationError(
                    f"Title must be a string, got {type(title)}",
                    [
                        "Convert title to string: str(title)",
                        "Use descriptive string for plot title"
                    ]
                )
            if len(title.strip()) == 0:
                raise ValidationError(
                    "Title cannot be empty",
                    [
                        "Provide descriptive title for plot",
                        "Use meaningful title that describes the visualization"
                    ]
                )
        
        # Validate figsize
        if 'figsize' in kwargs and kwargs['figsize'] is not None:
            figsize = kwargs['figsize']
            if isinstance(figsize, (list, tuple)):
                if len(figsize) != 2:
                    raise ValidationError(
                        f"figsize must have 2 elements, got {len(figsize)}",
                        [
                            "Use format: figsize=(width, height)",
                            "Example: figsize=(10, 8)"
                        ]
                    )
                if not all(isinstance(x, (int, float)) for x in figsize):
                    raise ValidationError(
                        "figsize elements must be numbers",
                        [
                            "Use numeric values: figsize=(10, 8)",
                            "Check figsize parameter types"
                        ]
                    )
    
    @staticmethod
    def validate_with_graceful_fallback(validation_func, 
                                      *args, 
                                      fallback_value=None,
                                      **kwargs) -> Any:
        """
        Run validation with graceful fallback.
        
        Args:
            validation_func: Validation function to run
            *args: Arguments for validation function
            fallback_value: Value to return if validation fails
            **kwargs: Keyword arguments for validation function
            
        Returns:
            Original value if validation passes, fallback_value if it fails
        """
        try:
            validation_func(*args, **kwargs)
            return args[0] if args else None
        except ValidationError as e:
            logger.warning(f"Validation failed, using fallback: {e.message}")
            return fallback_value
    
    @staticmethod
    def suggest_fixes(error_type: str, context: Dict[str, Any]) -> List[str]:
        """
        Generate context-aware fix suggestions.
        
        Args:
            error_type: Type of error encountered
            context: Context information about the error
            
        Returns:
            List of suggested fixes
        """
        suggestions = []
        
        if error_type == "shape_mismatch":
            data_shape = context.get('actual_shape')
            expected_shape = context.get('expected_shape')
            data_name = context.get('data_name', 'data')
            
            if data_shape and expected_shape:
                if len(data_shape) == len(expected_shape):
                    suggestions.append(f"Reshape {data_name}: {data_name}.reshape{expected_shape}")
                else:
                    suggestions.append(f"Check data preprocessing for {data_name}")
                    suggestions.append(f"Verify data loading pipeline")
        
        elif error_type == "missing_method":
            method_name = context.get('method_name')
            if method_name:
                suggestions.append(f"Implement {method_name}() method in your model")
                suggestions.append(f"Check model training and initialization")
        
        elif error_type == "invalid_data_type":
            data_type = context.get('actual_type')
            expected_type = context.get('expected_type')
            if data_type and expected_type:
                suggestions.append(f"Convert to {expected_type}: {expected_type}(data)")
        
        return suggestions


# Convenience functions for common validations
def validate_visualization_inputs(features: np.ndarray,
                                labels: np.ndarray,
                                model: Any = None) -> None:
    """
    Validate common inputs for visualization functions.
    
    Args:
        features: Input features
        labels: Target labels
        model: Model object (optional)
        
    Raises:
        ValidationError: If inputs are invalid
    """
    validator = VisualizationValidator()
    
    # Validate features
    validator.validate_2d_features(features, "features")
    
    # Validate labels
    validator.validate_labels(labels, data_name="labels")
    
    # Validate model if provided
    if model is not None:
        validator.validate_model_interface(model)
    
    # Validate matching lengths
    if len(features) != len(labels):
        raise ValidationError(
            f"Features and labels must have same length: {len(features)} vs {len(labels)}",
            [
                "Check data loading and preprocessing",
                "Ensure features and labels are aligned",
                "Verify data splitting and shuffling"
            ]
        )


def validate_training_visualization_inputs(training_data: Dict[str, List[float]],
                                         model: Any = None) -> None:
    """
    Validate inputs for training visualization functions.
    
    Args:
        training_data: Training metrics dictionary
        model: Model object (optional)
        
    Raises:
        ValidationError: If inputs are invalid
    """
    validator = VisualizationValidator()
    
    # Validate training data
    validator.validate_training_data(training_data)
    
    # Validate model if provided
    if model is not None:
        validator.validate_model_interface(model)


def create_validation_error_with_context(error_type: str,
                                       message: str,
                                       context: Dict[str, Any]) -> ValidationError:
    """
    Create a ValidationError with context-aware suggestions.
    
    Args:
        error_type: Type of error
        message: Error message
        context: Context information
        
    Returns:
        ValidationError with suggestions
    """
    suggestions = VisualizationValidator.suggest_fixes(error_type, context)
    return ValidationError(message, suggestions) 