"""
AI From Scratch Shared Utilities

Provides standardized W&B integration patterns and common utilities
for the AI-From-Scratch-to-Scale educational project.

This package contains:
- BaseWandbVisualizer: Abstract base class for model-specific W&B integration
- Utility functions for W&B initialization and cleanup
- Common patterns for experiment tracking across all models

Example:
    >>> from ai_from_scratch_shared import BaseWandbVisualizer
    >>> class MyModelVisualizer(BaseWandbVisualizer):
    ...     def log_model_specific_metrics(self, metrics):
    ...         # Implementation here
    ...         pass
"""

from .wandb_integration import BaseWandbVisualizer, initialize_wandb, finish_wandb

__version__ = "1.0.0"
__author__ = "AI-From-Scratch-to-Scale Project"

__all__ = [
    "BaseWandbVisualizer",
    "initialize_wandb", 
    "finish_wandb",
]
