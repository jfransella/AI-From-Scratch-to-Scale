"""
AI From Scratch to Scale - Shared Utilities
==========================================

This package provides shared utilities for the AI-From-Scratch-to-Scale project,
including visualization frameworks, W&B integration, and common components.

Key Components:
- BaseVisualizer: Unified visualization interface
- PlotFactory: Standardized plot creation
- BaseWandbVisualizer: W&B integration base class
- InteractiveVisualizer: Real-time interactive plots
- AdvancedVisualizer: Advanced visualization types
"""

# Import visualization framework
from .visualization import (
    BaseVisualizer,
    PlotFactory,
    InteractiveVisualizer,
    AdvancedVisualizer
)

# Import W&B integration
from .wandb_integration import BaseWandbVisualizer

# Import utility functions
from .wandb_integration import initialize_wandb, finish_wandb

__version__ = "1.0.0"
__author__ = "AI-From-Scratch-to-Scale Team"

__all__ = [
    "BaseVisualizer",
    "PlotFactory", 
    "InteractiveVisualizer",
    "AdvancedVisualizer",
    "BaseWandbVisualizer",
    "initialize_wandb",
    "finish_wandb"
] 