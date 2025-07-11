"""
Visualization utilities for ADALINE (Adaptive Linear Neuron).

This module provides comprehensive visualization functions for ADALINE models
including training progress, decision boundaries, and model analysis plots.
"""

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, Union
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src.config import config, STYLE_SETTINGS, COLOR_PALETTE


logger = logging.getLogger(__name__)


class ADALINEVisualizer:
    """
    Comprehensive visualizer for ADALINE models.
    
    This class provides various visualization functions for understanding
    ADALINE training dynamics and model behavior.
    """
    
    def __init__(self, cfg: Any = None) -> None:
        """
        Initialize ADALINE visualizer.
        
        Args:
            cfg: Configuration object
        """
        self.config = cfg if cfg is not None else config
        self._setup_plotting_style()
        logger.info("ADALINE Visualizer initialized")
    
    def _setup_plotting_style(self) -> None:
        """Setup matplotlib plotting style."""
        plt.style.use('default')
        for key, value in STYLE_SETTINGS.items():
            plt.rcParams[key] = value
    
    def plot_training_progress(self, 
                              training_state: Any,
                              save: bool = True) -> str:
        """
        Plot training progress including loss curves.
        
        Args:
            training_state: Training state object containing history
            save: Whether to save the plot
            
        Returns:
            Path to saved plot file
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ADALINE Training Progress', fontsize=16, fontweight='bold')
        
        epochs = np.arange(1, len(training_state.training_loss) + 1)
        
        # Training loss
        axes[0, 0].plot(epochs, training_state.training_loss, 
                        color=COLOR_PALETTE[0], linewidth=2, label='Training Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Validation loss (if available)
        if len(training_state.validation_loss) > 0:
            axes[0, 1].plot(epochs, training_state.validation_loss, 
                           color=COLOR_PALETTE[1], linewidth=2, label='Validation Loss')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # Loss comparison
        if len(training_state.validation_loss) > 0:
            axes[1, 0].plot(epochs, training_state.training_loss, 
                           color=COLOR_PALETTE[0], linewidth=2, label='Training')
            axes[1, 0].plot(epochs, training_state.validation_loss, 
                           color=COLOR_PALETTE[1], linewidth=2, label='Validation')
            axes[1, 0].set_title('Training vs Validation Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # Loss improvement
        if len(training_state.training_loss) > 1:
            loss_improvement = np.diff(training_state.training_loss)
            axes[1, 1].plot(epochs[1:], loss_improvement, 
                           color=COLOR_PALETTE[2], linewidth=2)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Loss Improvement per Epoch')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Change')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.config.PLOTS_SAVE_PATH, 'training_progress.png')
            plt.savefig(plot_path, dpi=self.config.PLOT_DPI, bbox_inches='tight')
            logger.info(f"Training progress plot saved to: {plot_path}")
            plt.close()
            return plot_path
        else:
            plt.show()
            return ""
    
    def plot_weight_evolution(self, 
                             training_state: Any,
                             save: bool = True) -> str:
        """
        Plot weight evolution during training.
        
        Args:
            training_state: Training state object containing weight history
            save: Whether to save the plot
            
        Returns:
            Path to saved plot file
        """
        weight_history = training_state.weight_history
        bias_history = training_state.bias_history
        
        n_weights = weight_history.shape[1]
        epochs = np.arange(1, len(weight_history) + 1)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('ADALINE Weight Evolution', fontsize=16, fontweight='bold')
        
        # Weight evolution
        for i in range(n_weights):
            axes[0].plot(epochs, weight_history[:, i], 
                        color=COLOR_PALETTE[i % len(COLOR_PALETTE)], 
                        linewidth=2, label=f'Weight {i+1}')
        
        axes[0].set_title('Weight Evolution')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Weight Value')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Bias evolution
        axes[1].plot(epochs, bias_history, 
                    color=COLOR_PALETTE[-1], linewidth=2, label='Bias')
        axes[1].set_title('Bias Evolution')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Bias Value')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.config.PLOTS_SAVE_PATH, 'weight_evolution.png')
            plt.savefig(plot_path, dpi=self.config.PLOT_DPI, bbox_inches='tight')
            logger.info(f"Weight evolution plot saved to: {plot_path}")
            plt.close()
            return plot_path
        else:
            plt.show()
            return ""
    
    def plot_decision_boundary(self, 
                              model: Any,
                              data_splits: Dict[str, np.ndarray],
                              save: bool = True) -> str:
        """
        Plot decision boundary for 2D data.
        
        Args:
            model: Trained ADALINE model
            data_splits: Dictionary containing data splits
            save: Whether to save the plot
            
        Returns:
            Path to saved plot file
        """
        # Only works for 2D data
        if data_splits['X_train'].shape[1] != 2:
            logger.warning("Decision boundary plot only works for 2D data")
            return ""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('ADALINE Decision Boundary', fontsize=16, fontweight='bold')
        
        # Create meshgrid for decision boundary
        X_train = data_splits['X_train']
        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # Make predictions on meshgrid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        if self.config.ADD_BIAS_TERM:
            grid_points = np.hstack([np.ones((grid_points.shape[0], 1)), grid_points])
        
        Z = model.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # Plot training data
        axes[0].scatter(data_splits['X_train'][:, 0], data_splits['X_train'][:, 1], 
                       c=data_splits['y_train'].flatten(), cmap='viridis', 
                       alpha=0.7, s=50)
        axes[0].contour(xx, yy, Z, levels=[0], colors='red', linewidths=2)
        axes[0].set_title('Training Data with Decision Boundary')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        axes[0].grid(True, alpha=0.3)
        
        # Plot test data
        axes[1].scatter(data_splits['X_test'][:, 0], data_splits['X_test'][:, 1], 
                       c=data_splits['y_test'].flatten(), cmap='viridis', 
                       alpha=0.7, s=50)
        axes[1].contour(xx, yy, Z, levels=[0], colors='red', linewidths=2)
        axes[1].set_title('Test Data with Decision Boundary')
        axes[1].set_xlabel('Feature 1')
        axes[1].set_ylabel('Feature 2')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.config.PLOTS_SAVE_PATH, 'decision_boundary.png')
            plt.savefig(plot_path, dpi=self.config.PLOT_DPI, bbox_inches='tight')
            logger.info(f"Decision boundary plot saved to: {plot_path}")
            plt.close()
            return plot_path
        else:
            plt.show()
            return ""
    
    def plot_model_comparison(self, 
                             model: Any,
                             data_splits: Dict[str, np.ndarray],
                             save: bool = True) -> str:
        """
        Plot model comparison and analysis.
        
        Args:
            model: Trained ADALINE model
            data_splits: Dictionary containing data splits
            save: Whether to save the plot
            
        Returns:
            Path to saved plot file
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ADALINE Model Analysis', fontsize=16, fontweight='bold')
        
        # Predictions vs actual
        y_train_pred = model.predict(data_splits['X_train'])
        y_test_pred = model.predict(data_splits['X_test'])
        
        # Training predictions
        axes[0, 0].scatter(data_splits['y_train'].flatten(), y_train_pred, 
                           alpha=0.6, color=COLOR_PALETTE[0])
        axes[0, 0].plot([data_splits['y_train'].min(), data_splits['y_train'].max()], 
                        [data_splits['y_train'].min(), data_splits['y_train'].max()], 
                        'r--', linewidth=2)
        axes[0, 0].set_title('Training: Predicted vs Actual')
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Test predictions
        axes[0, 1].scatter(data_splits['y_test'].flatten(), y_test_pred, 
                           alpha=0.6, color=COLOR_PALETTE[1])
        axes[0, 1].plot([data_splits['y_test'].min(), data_splits['y_test'].max()], 
                        [data_splits['y_test'].min(), data_splits['y_test'].max()], 
                        'r--', linewidth=2)
        axes[0, 1].set_title('Test: Predicted vs Actual')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Predicted Values')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals
        train_residuals = data_splits['y_train'].flatten() - y_train_pred
        test_residuals = data_splits['y_test'].flatten() - y_test_pred
        
        axes[1, 0].hist(train_residuals, bins=20, alpha=0.7, color=COLOR_PALETTE[0], 
                        label='Training')
        axes[1, 0].hist(test_residuals, bins=20, alpha=0.7, color=COLOR_PALETTE[1], 
                        label='Test')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals vs predicted
        axes[1, 1].scatter(y_train_pred, train_residuals, alpha=0.6, 
                           color=COLOR_PALETTE[0], label='Training')
        axes[1, 1].scatter(y_test_pred, test_residuals, alpha=0.6, 
                           color=COLOR_PALETTE[1], label='Test')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Residuals vs Predicted Values')
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.config.PLOTS_SAVE_PATH, 'model_comparison.png')
            plt.savefig(plot_path, dpi=self.config.PLOT_DPI, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to: {plot_path}")
            plt.close()
            return plot_path
        else:
            plt.show()
            return ""
    
    def plot_learning_curves(self, 
                            training_state: Any,
                            save: bool = True) -> str:
        """
        Plot learning curves for model analysis.
        
        Args:
            training_state: Training state object
            save: Whether to save the plot
            
        Returns:
            Path to saved plot file
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('ADALINE Learning Curves', fontsize=16, fontweight='bold')
        
        epochs = np.arange(1, len(training_state.training_loss) + 1)
        
        # Loss curves
        axes[0].plot(epochs, training_state.training_loss, 
                    color=COLOR_PALETTE[0], linewidth=2, label='Training Loss')
        if len(training_state.validation_loss) > 0:
            axes[0].plot(epochs, training_state.validation_loss, 
                        color=COLOR_PALETTE[1], linewidth=2, label='Validation Loss')
        axes[0].set_title('Loss Curves')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate analysis
        if len(training_state.training_loss) > 1:
            loss_gradient = np.gradient(training_state.training_loss)
            axes[1].plot(epochs[1:], loss_gradient[1:], 
                        color=COLOR_PALETTE[2], linewidth=2)
            axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1].set_title('Loss Gradient')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss Gradient')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.config.PLOTS_SAVE_PATH, 'learning_curves.png')
            plt.savefig(plot_path, dpi=self.config.PLOT_DPI, bbox_inches='tight')
            logger.info(f"Learning curves plot saved to: {plot_path}")
            plt.close()
            return plot_path
        else:
            plt.show()
            return ""


def create_visualizer(config: Any = None) -> ADALINEVisualizer:
    """
    Factory function to create an ADALINE visualizer.
    
    Args:
        config: Configuration object
        
    Returns:
        ADALINEVisualizer instance
    """
    return ADALINEVisualizer(config) 