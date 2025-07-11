"""
AI From Scratch Shared Utilities

Provides standardized W&B integration patterns and common utilities
for the AI-From-Scratch-to-Scale educational project.

This package contains:
- BaseWandbVisualizer: Abstract base class for model-specific W&B integration
- Utility functions for W&B initialization and cleanup
- Common patterns for experiment tracking across all models
- Visualization framework: Standardized visualization components and styling

Example:
    >>> from ai_from_scratch_shared import BaseWandbVisualizer
    >>> from ai_from_scratch_shared.visualization import BaseVisualizer
    >>> class MyModelVisualizer(BaseVisualizer):
    ...     def plot_model_specific_data(self, data):
    ...         # Implementation here
    ...         pass
"""

from .wandb_integration import BaseWandbVisualizer, initialize_wandb, finish_wandb
# Note: visualization module is imported separately due to its size

__version__ = "1.0.0"
__author__ = "AI-From-Scratch-to-Scale Project"

__all__ = [
    "BaseWandbVisualizer",
    "initialize_wandb", 
    "finish_wandb",
]
