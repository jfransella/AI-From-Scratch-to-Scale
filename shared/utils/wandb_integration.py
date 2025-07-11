"""
Base Weights & Biases Integration for AI-From-Scratch-to-Scale
============================================================

This module provides a reusable base class for W&B experiment tracking
that can be extended by model-specific implementations across the project.

Educational Objectives:
- Demonstrate professional ML experiment tracking patterns
- Show clean separation of concerns in software architecture
- Provide consistent interfaces across different model types
- Enable systematic comparison of ML experiments

Key Features:
- Optional W&B integration (graceful fallback when disabled)
- Standardized logging interfaces for metrics, images, and artifacts
- Error handling and dependency management
- Abstract methods for model-specific customization
- Professional ML development workflow patterns
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)


class BaseWandbVisualizer(ABC):
    """
    Base class for Weights & Biases integration across all models.
    
    This class provides a standardized interface for experiment tracking
    while allowing model-specific customization through abstract methods.
    
    Educational Focus:
    - Demonstrates inheritance and composition patterns
    - Shows professional ML experiment tracking
    - Provides consistent interfaces across models
    - Enables systematic hyperparameter exploration
    
    Attributes:
        enabled (bool): Whether W&B logging is active
        wandb_run: Active W&B run object (if enabled)
        plots_dir (str): Directory for saving plots locally
    """
    
    def __init__(self, 
                 wandb_run: Optional[Any] = None, 
                 enabled: bool = True,
                 plots_dir: str = "outputs/plots") -> None:
        """
        Initialize the base W&B visualizer.
        
        Args:
            wandb_run: Active Weights & Biases run object
            enabled: Whether to enable W&B logging
            plots_dir: Directory for saving plots locally
            
        Raises:
            ImportError: If wandb is not installed but enabled=True
            ValueError: If wandb_run is None when enabled=True
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.wandb_run = wandb_run
        self.plots_dir = plots_dir
        
        # Create plots directory
        Path(plots_dir).mkdir(parents=True, exist_ok=True)
        
        # Check W&B availability and validate
        self._check_wandb_availability(enabled)
        self._validate_initialization()
        
        # Log initialization status
        if self.enabled:
            logger.info("W&B visualizer initialized - experiment tracking enabled")
        else:
            logger.info("W&B visualizer initialized - local logging only")
    
    def _check_wandb_availability(self, requested_enabled: bool) -> None:
        """
        Check if W&B is available and handle graceful fallback.
        
        Args:
            requested_enabled: Whether user requested W&B to be enabled
        """
        if requested_enabled and not WANDB_AVAILABLE:
            logger.warning(
                "Weights & Biases not available. Install with: pip install wandb\n"
                "Continuing with local logging only..."
            )
            self.enabled = False
    
    def _validate_initialization(self) -> None:
        """
        Validate that initialization parameters are consistent.
        
        Raises:
            ValueError: If wandb_run is None when W&B is enabled
        """
        if self.enabled and self.wandb_run is None:
            raise ValueError(
                "wandb_run cannot be None when visualization is enabled. "
                "Initialize wandb.init() first or set enabled=False"
            )
    
    # =================================================================
    # CORE LOGGING METHODS (REUSABLE ACROSS ALL MODELS)
    # =================================================================
    
    def _log_metrics(self, metrics: Dict[str, Union[int, float]], 
                    step: Optional[int] = None) -> None:
        """
        Internal method to log metrics to W&B.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for time series logging
        """
        if self.enabled:
            self.wandb_run.log(metrics, step=step)
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], 
                    step: Optional[int] = None) -> None:
        """
        Log metrics to W&B and provide local logging fallback.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for time series logging
            
        Educational Focus:
        Shows how to provide consistent interfaces while handling
        optional dependencies gracefully.
        """
        if self.enabled:
            self._log_metrics(metrics, step)
        else:
            # Local logging fallback
            step_info = f" (step {step})" if step is not None else ""
            logger.info(f"Metrics{step_info}: {metrics}")
    
    def log_image(self, image_path: str, key: str, caption: str = "") -> None:
        """
        Log an image file to W&B.
        
        Args:
            image_path: Path to the image file
            key: W&B logging key for the image
            caption: Optional caption for the image
            
        Educational Focus:
        Demonstrates file-based logging patterns common in ML workflows.
        """
        if not self.enabled:
            logger.debug(f"Image logging disabled: {key} -> {image_path}")
            return
        
        try:
            # Log the image to W&B
            image_obj = wandb.Image(image_path, caption=caption)
            self.wandb_run.log({key: image_obj})
            logger.info(f"Image logged to W&B: {key}")
            
        except Exception as e:
            logger.warning(f"Failed to log image to W&B: {e}")
    
    def log_figure(self, figure: matplotlib.figure.Figure, name: str, 
                  step: Optional[int] = None, close_figure: bool = True) -> None:
        """
        Log a matplotlib figure to W&B and save locally.
        
        Args:
            figure: Matplotlib figure to log
            name: Name for the figure (used for filename and W&B key)
            step: Optional step number for time series logging
            close_figure: Whether to close the figure after logging
            
        Educational Focus:
        Shows best practices for handling matplotlib figures in ML pipelines:
        - Always save locally for reproducibility
        - Clean up memory by closing figures
        - Handle optional remote logging gracefully
        """
        # Always save locally for reproducibility
        local_path = os.path.join(self.plots_dir, f"{name}.png")
        figure.savefig(local_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved locally: {local_path}")
        
        # Log to W&B if enabled
        if self.enabled:
            try:
                self.wandb_run.log({f"plots/{name}": wandb.Image(figure)}, step=step)
                logger.info(f"Figure logged to W&B: {name}")
            except Exception as e:
                logger.warning(f"Failed to log figure to W&B: {e}")
        
        # Clean up memory
        if close_figure:
            plt.close(figure)
    
    def save_model_artifact(self, model_state: Dict[str, Any], 
                           artifact_name: str) -> None:
        """
        Save model state as W&B artifact with local backup.
        
        Args:
            model_state: Dictionary containing model state
            artifact_name: Name for the artifact
            
        Educational Focus:
        Demonstrates ML model versioning and artifact management patterns.
        """
        # Always save locally
        local_path = os.path.join(self.plots_dir, f"{artifact_name}.npz")
        np.savez_compressed(local_path, **model_state)
        logger.info(f"Model artifact saved locally: {local_path}")
        
        if not self.enabled:
            return
        
        try:
            # Create W&B artifact
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"Model state and configuration for {artifact_name}"
            )
            
            # Add file to artifact
            artifact.add_file(local_path)
            
            # Log artifact
            self.wandb_run.log_artifact(artifact)
            logger.info(f"Model artifact logged to W&B: {artifact_name}")
            
        except Exception as e:
            logger.warning(f"Failed to log model artifact to W&B: {e}")
    
    def log_file_artifact(self, file_path: str, artifact_name: str, 
                         description: str = "") -> None:
        """
        Log an existing file as an artifact to W&B.
        
        Args:
            file_path: Path to the file to log
            artifact_name: Name for the artifact
            description: Optional description of the artifact
        """
        if not self.enabled:
            logger.info(f"File artifact logged locally: {artifact_name} at {file_path}")
            return
        
        try:
            # Create artifact
            artifact = wandb.Artifact(
                name=artifact_name,
                type="dataset" if "report" in artifact_name.lower() else "file",
                description=description
            )
            
            # Add file to artifact
            artifact.add_file(file_path)
            
            # Log artifact
            self.wandb_run.log_artifact(artifact)
            logger.info(f"File artifact logged to W&B: {artifact_name}")
            
        except Exception as e:
            logger.warning(f"Failed to log file artifact to W&B: {e}")
    
    def log_experiment_results(self, experiment_name: str, 
                             results: Dict[str, Any], 
                             step: Optional[int] = None) -> None:
        """
        Log comprehensive experiment results with intelligent metric processing.
        
        Args:
            experiment_name: Name of the experiment
            results: Dictionary containing experiment results
            step: Optional step number for time series logging
            
        Educational Focus:
        Shows how to handle complex nested data structures and extract
        meaningful metrics for experiment tracking.
        """
        metrics = {}
        
        # Process results dictionary intelligently
        for key, value in results.items():
            metric_key = f"{experiment_name}/{key}"
            
            if isinstance(value, (int, float)):
                metrics[metric_key] = value
            elif isinstance(value, dict):
                # Flatten nested dictionaries
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        metrics[f"{metric_key}_{subkey}"] = subvalue
            elif isinstance(value, (list, np.ndarray)):
                # Log statistical summaries for arrays
                try:
                    array_vals = np.array(value)
                    if array_vals.dtype != 'object' and len(array_vals) > 0:
                        metrics[f"{metric_key}_mean"] = np.mean(array_vals)
                        metrics[f"{metric_key}_std"] = np.std(array_vals)
                        metrics[f"{metric_key}_min"] = np.min(array_vals)
                        metrics[f"{metric_key}_max"] = np.max(array_vals)
                    else:
                        metrics[f"{metric_key}_count"] = len(value)
                except (ValueError, TypeError):
                    metrics[f"{metric_key}_count"] = len(value)
        
        # Log processed metrics
        self.log_metrics(metrics, step=step)
        logger.info(f"Experiment results logged: {experiment_name}")
    
    def create_experiment_summary(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Create a comprehensive experiment summary table.
        
        Args:
            all_results: Dictionary of all experiment results
            
        Educational Focus:
        Demonstrates how to create comprehensive experiment comparisons
        for systematic analysis and reporting.
        """
        if not self.enabled:
            logger.info("Experiment summary created locally (W&B disabled)")
            return
        
        try:
            # Create summary table data
            table_data = []
            for exp_name, results in all_results.items():
                row = [exp_name]
                
                # Extract key metrics (customize per model type)
                metrics = self._extract_summary_metrics(results)
                row.extend(metrics)
                table_data.append(row)
            
            # Create W&B table
            columns = ["Experiment"] + self._get_summary_columns()
            table = wandb.Table(columns=columns, data=table_data)
            
            self.wandb_run.log({"experiment_summary": table})
            logger.info("Experiment summary table created")
            
        except Exception as e:
            logger.warning(f"Failed to create experiment summary: {e}")
    
    def _extract_summary_metrics(self, results: Dict[str, Any]) -> List[str]:
        """
        Extract key metrics for summary table (to be overridden by subclasses).
        
        Args:
            results: Experiment results dictionary
            
        Returns:
            List of formatted metric values
        """
        # Default implementation - subclasses should override
        return ["N/A"] * len(self._get_summary_columns())
    
    def _get_summary_columns(self) -> List[str]:
        """
        Get column names for summary table (to be overridden by subclasses).
        
        Returns:
            List of column names
        """
        # Default implementation - subclasses should override
        return ["Status"]
    
    # =================================================================
    # ABSTRACT METHODS (MUST BE IMPLEMENTED BY MODEL-SPECIFIC CLASSES)
    # =================================================================
    
    @abstractmethod
    def log_model_config(self, config: Dict[str, Any]) -> None:
        """
        Log model-specific configuration and hyperparameters.
        
        Args:
            config: Model configuration dictionary
            
        Educational Focus:
        Each model type has different configuration parameters that
        need to be tracked for reproducibility and comparison.
        """
        pass
    
    @abstractmethod
    def log_training_progress(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log model-specific training progress metrics.
        
        Args:
            metrics: Training metrics dictionary
            step: Training step/epoch number
            
        Educational Focus:
        Different models have different training dynamics that need
        to be monitored (loss curves, convergence, etc.).
        """
        pass
    
    @abstractmethod
    def create_model_visualizations(self, **kwargs) -> None:
        """
        Create model-specific visualizations and analysis plots.
        
        Args:
            **kwargs: Model-specific parameters
            
        Educational Focus:
        Each model type has unique visualizations that help understand
        its behavior (decision boundaries, energy landscapes, etc.).
        """
        pass


# =================================================================
# UTILITY FUNCTIONS (REUSABLE ACROSS ALL MODELS)
# =================================================================

def initialize_wandb(project_name: str, 
                    entity: Optional[str] = None,
                    config: Optional[Dict[str, Any]] = None,
                    enabled: bool = True) -> Tuple[Any, BaseWandbVisualizer]:
    """
    Initialize Weights & Biases run and create a base visualizer.
    
    Args:
        project_name: W&B project name
        entity: W&B entity (username or team)
        config: Configuration dictionary to log
        enabled: Whether to enable W&B logging
    
    Returns:
        Tuple of (wandb_run, base_visualizer)
        
    Educational Focus:
    Shows standardized initialization patterns for ML experiments
    with proper error handling and fallback mechanisms.
    """
    if not enabled or not WANDB_AVAILABLE:
        logger.info("W&B integration disabled")
        return None, BaseWandbVisualizer(enabled=False)
    
    try:
        # Initialize W&B run
        wandb_run = wandb.init(
            project=project_name,
            entity=entity,
            config=config or {},
            mode="online" if enabled else "disabled"
        )
        
        # Note: We can't instantiate BaseWandbVisualizer directly since it's abstract
        # Model-specific implementations will create their own visualizers
        logger.info(f"W&B run initialized: {wandb_run.name}")
        return wandb_run, None
        
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}. Continuing with local logging only.")
        return None, BaseWandbVisualizer(enabled=False)


def finish_wandb(wandb_run: Optional[Any]) -> None:
    """
    Finish W&B run gracefully.
    
    Args:
        wandb_run: W&B run object to finish
        
    Educational Focus:
    Shows proper cleanup patterns for ML experiments.
    """
    if wandb_run is not None:
        try:
            wandb_run.finish()
            logger.info("W&B run finished successfully")
        except Exception as e:
            logger.warning(f"Error finishing W&B run: {e}")
