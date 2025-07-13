"""
Shared Weights & Biases Integration Framework
===========================================

This module provides the BaseWandbVisualizer class that all model-specific
W&B visualizers inherit from. It establishes consistent patterns for ML
experiment tracking across the AI-From-Scratch-to-Scale project.

Key Features:
- Abstract base class for model-specific W&B integration
- Consistent error handling and graceful fallbacks
- Standardized logging patterns
- Educational focus with comprehensive documentation
- Professional ML experiment tracking practices

Educational Objectives:
- Understand inheritance and abstract base classes
- Learn professional ML experiment tracking patterns
- Practice separation of concerns in software architecture
- Demonstrate consistent API design across models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import wandb, but provide graceful fallback
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available. W&B logging will be disabled.")

logger = logging.getLogger(__name__)


class BaseWandbVisualizer(ABC):
    """
    Abstract base class for model-specific W&B integration.
    
    This class provides the foundation for consistent W&B experiment tracking
    across all models in the AI-From-Scratch-to-Scale project. It handles
    common functionality like initialization, error handling, and basic logging
    while requiring model-specific implementations for specialized features.
    
    Features:
    - Automatic W&B availability checking
    - Graceful fallback when W&B is unavailable
    - Consistent error handling and logging
    - Standardized initialization patterns
    - Abstract methods for model-specific implementations
    
    Example:
        class PerceptronWandbVisualizer(BaseWandbVisualizer):
            def log_model_config(self, config):
                # Model-specific implementation
                pass
            
            def log_training_progress(self, metrics, step):
                # Model-specific implementation
                pass
    """
    
    def __init__(self, wandb_run: Optional[Any] = None, enabled: bool = True) -> None:
        """
        Initialize the base W&B visualizer.
        
        Args:
            wandb_run: Active Weights & Biases run object
            enabled: Whether to enable W&B logging
        """
        self.wandb_run = wandb_run
        self.enabled = enabled and WANDB_AVAILABLE
        
        # Validate initialization
        self._validate_initialization()
        
        if self.enabled:
            logger.info("W&B visualizer initialized - experiment tracking enabled")
        else:
            logger.info("W&B visualizer initialized - local logging only")
    
    def _validate_initialization(self) -> None:
        """Validate the initialization parameters."""
        if self.enabled and not WANDB_AVAILABLE:
            logger.warning("W&B requested but not available. Disabling W&B logging.")
            self.enabled = False
        
        if self.enabled and self.wandb_run is None:
            logger.warning("W&B enabled but no run provided. Disabling W&B logging.")
            self.enabled = False
    
    def _log_metrics(self, metrics: Dict[str, Union[int, float]], 
                    step: Optional[int] = None) -> None:
        """
        Log metrics to W&B with error handling.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number for the metrics
        """
        if not self.enabled:
            logger.debug(f"Skipped logging metrics: {list(metrics.keys())}")
            return
        
        try:
            self.wandb_run.log(metrics, step=step)
            logger.debug(f"Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], 
                   step: Optional[int] = None) -> None:
        """
        Public interface for logging metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number for the metrics
        """
        self._log_metrics(metrics, step)
    
    def log_figure(self, figure: plt.Figure, name: str, 
                  step: Optional[int] = None, close_figure: bool = True) -> None:
        """
        Log a matplotlib figure to W&B.
        
        Args:
            figure: Matplotlib figure to log
            name: Name for the figure in W&B
            step: Optional step number
            close_figure: Whether to close the figure after logging
        """
        if not self.enabled:
            logger.debug(f"Skipped logging figure: {name}")
            if close_figure:
                plt.close(figure)
            return
        
        try:
            self.wandb_run.log({name: wandb.Image(figure)}, step=step)
            logger.debug(f"Logged figure: {name}")
        except Exception as e:
            logger.error(f"Failed to log figure '{name}': {e}")
        finally:
            if close_figure:
                plt.close(figure)
    
    def log_figure_with_metadata(self, 
                                figure: plt.Figure,
                                name: str,
                                plot_type: str,
                                model_info: Optional[Dict[str, Any]] = None,
                                dataset_info: Optional[Dict[str, Any]] = None,
                                hyperparameters: Optional[Dict[str, Any]] = None,
                                step: Optional[int] = None,
                                close_figure: bool = True) -> None:
        """
        Log a figure with comprehensive metadata.
        
        Args:
            figure: Matplotlib figure to log
            name: Name for the figure in W&B
            plot_type: Type of plot for categorization
            model_info: Optional model information
            dataset_info: Optional dataset information
            hyperparameters: Optional hyperparameters
            step: Optional step number
            close_figure: Whether to close the figure after logging
        """
        if not self.enabled:
            logger.debug(f"Skipped logging figure with metadata: {name}")
            if close_figure:
                plt.close(figure)
            return
        
        try:
            # Generate caption with metadata
            caption = self._generate_plot_caption(plot_type, model_info, dataset_info, hyperparameters)
            
            # Extract metadata for logging
            metadata = self._extract_plot_metadata(plot_type, model_info, dataset_info, hyperparameters, step)
            
            # Log figure with metadata
            self.wandb_run.log({
                name: wandb.Image(figure, caption=caption),
                f"{name}_metadata": metadata
            }, step=step)
            
            logger.debug(f"Logged figure with metadata: {name}")
            
        except Exception as e:
            logger.error(f"Failed to log figure with metadata '{name}': {e}")
        finally:
            if close_figure:
                plt.close(figure)
    
    def _generate_plot_caption(self, 
                              plot_type: str,
                              model_info: Optional[Dict[str, Any]] = None,
                              dataset_info: Optional[Dict[str, Any]] = None,
                              hyperparameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate an educational caption for the plot.
        
        Args:
            plot_type: Type of plot
            model_info: Optional model information
            dataset_info: Optional dataset information
            hyperparameters: Optional hyperparameters
            
        Returns:
            Generated caption string
        """
        caption_parts = [f"{plot_type.replace('_', ' ').title()} showing classification performance"]
        
        if model_info:
            model_name = model_info.get('model_type', 'Model')
            caption_parts.append(f"Model: {model_name}")
        
        if dataset_info:
            dataset_name = dataset_info.get('name', 'Dataset')
            n_samples = dataset_info.get('n_samples', 0)
            caption_parts.append(f"Dataset: {dataset_name} ({n_samples} samples)")
        
        if hyperparameters:
            # Format key hyperparameters
            key_params = ['learning_rate', 'epochs', 'optimizer']
            param_strs = []
            for param in key_params:
                if param in hyperparameters:
                    value = hyperparameters[param]
                    if isinstance(value, float):
                        param_strs.append(f"{param}={value:.4f}")
                    else:
                        param_strs.append(f"{param}={value}")
            
            if param_strs:
                caption_parts.append(f"Parameters: {', '.join(param_strs)}")
        
        return " | ".join(caption_parts)
    
    def _extract_plot_metadata(self,
                               plot_type: str,
                               model_info: Optional[Dict[str, Any]] = None,
                               dataset_info: Optional[Dict[str, Any]] = None,
                               hyperparameters: Optional[Dict[str, Any]] = None,
                               step: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract metadata for the plot.
        
        Args:
            plot_type: Type of plot
            model_info: Optional model information
            dataset_info: Optional dataset information
            hyperparameters: Optional hyperparameters
            step: Optional step number
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            'plot_type': plot_type,
            'timestamp': getattr(self.wandb_run, 'start_time', None) if self.enabled and self.wandb_run is not None else None,
            'step': step
        }
        
        if model_info:
            metadata['model'] = model_info
        
        if dataset_info:
            metadata['dataset'] = dataset_info
        
        if hyperparameters:
            metadata['hyperparameters'] = hyperparameters
        
        return metadata
    
    def save_model_artifact(self, model_state: Dict[str, Any], artifact_name: str) -> None:
        """
        Save model state as a W&B artifact.
        
        Args:
            model_state: Model state dictionary
            artifact_name: Name for the artifact
        """
        if not self.enabled:
            logger.debug(f"Skipped saving model artifact: {artifact_name}")
            return
        
        try:
            artifact = wandb.Artifact(artifact_name, type="model")
            # Add model state to artifact
            # Implementation depends on model serialization method
            self.wandb_run.log_artifact(artifact)
            logger.debug(f"Saved model artifact: {artifact_name}")
        except Exception as e:
            logger.error(f"Failed to save model artifact '{artifact_name}': {e}")
    
    def log_file_artifact(self, file_path: str, artifact_name: str, 
                         description: str = "") -> None:
        """
        Log a file as a W&B artifact.
        
        Args:
            file_path: Path to the file
            artifact_name: Name for the artifact
            description: Optional description
        """
        if not self.enabled:
            logger.debug(f"Skipped logging file artifact: {artifact_name}")
            return
        
        try:
            artifact = wandb.Artifact(artifact_name, type="file", description=description)
            artifact.add_file(file_path)
            self.wandb_run.log_artifact(artifact)
            logger.debug(f"Logged file artifact: {artifact_name}")
        except Exception as e:
            logger.error(f"Failed to log file artifact '{artifact_name}': {e}")
    
    @abstractmethod
    def log_model_config(self, config: Dict[str, Any]) -> None:
        """
        Log model configuration (abstract method).
        
        Args:
            config: Model configuration dictionary
        """
        pass
    
    @abstractmethod
    def log_training_progress(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log training progress (abstract method).
        
        Args:
            metrics: Training metrics dictionary
            step: Current training step
        """
        pass
    
    @abstractmethod
    def create_model_visualizations(self, model: Any, features: Any, 
                                  y: Any, predictions: Any) -> Dict[str, Any]:
        """
        Create model-specific visualizations (abstract method).
        
        Args:
            model: Trained model
            features: Input features
            y: True labels
            predictions: Model predictions
            
        Returns:
            Dictionary of visualization results
        """
        pass


def initialize_wandb(project_name: str, 
                    entity: Optional[str] = None,
                    config: Optional[Dict[str, Any]] = None,
                    enabled: bool = True) -> Optional[Any]:
    """
    Initialize a W&B run with standard configuration.
    
    Args:
        project_name: Name of the W&B project
        entity: Optional W&B entity/username
        config: Optional configuration dictionary
        enabled: Whether to enable W&B logging
        
    Returns:
        W&B run object or None if disabled/unavailable
    """
    if not enabled or not WANDB_AVAILABLE:
        logger.info("W&B initialization skipped (disabled or unavailable)")
        return None
    
    try:
        run = wandb.init(
            project=project_name,
            entity=entity,
            config=config or {},
            mode="online" if enabled else "disabled"
        )
        logger.info(f"W&B run initialized: {run.name}")
        return run
    except Exception as e:
        logger.error(f"Failed to initialize W&B: {e}")
        return None


def finish_wandb(wandb_run: Optional[Any]) -> None:
    """
    Finish a W&B run with proper cleanup.
    
    Args:
        wandb_run: W&B run object to finish
    """
    if wandb_run is not None and WANDB_AVAILABLE:
        try:
            wandb_run.finish()
            logger.info("W&B run finished successfully")
        except Exception as e:
            logger.error(f"Failed to finish W&B run: {e}")
    else:
        logger.debug("W&B run finish skipped (no run or W&B unavailable)") 