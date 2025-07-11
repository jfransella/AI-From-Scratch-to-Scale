"""
Utility modules for the AI-From-Scratch-to-Scale project.

Contains reusable components for:
- W&B experiment tracking integration
- Common visualization utilities
- Data processing helpers
- Model evaluation tools
"""

from .wandb_integration import initialize_wandb, finish_wandb

__all__ = [
    "initialize_wandb", 
    "finish_wandb"
]

# Note: BaseWandbVisualizer is abstract and should be imported directly
# from wandb_integration module in model-specific implementations
