# AI From Scratch Shared Utilities

Shared utilities package for the AI-From-Scratch-to-Scale educational project.

## Overview

This package provides standardized Weights & Biases integration patterns and common utilities that can be used across all model implementations in the project.

## Features

- **BaseWandbVisualizer**: Abstract base class for model-specific W&B experiment tracking
- **Utility Functions**: Helper functions for W&B initialization and cleanup
- **Educational Focus**: Code designed for learning with extensive documentation
- **Professional Patterns**: Industry-standard ML experiment tracking practices

## Installation

Install the package in development mode from the project root:

```bash
pip install -e ai_from_scratch_shared/
```

Or add to your `requirements.txt`:

```
-e ../ai_from_scratch_shared
```

## Usage

### Basic Usage

```python
from ai_from_scratch_shared import BaseWandbVisualizer, initialize_wandb, finish_wandb

# Initialize W&B for your experiment
wandb_run = initialize_wandb(
    project="perceptron-experiments",
    name="my-experiment",
    config={"learning_rate": 0.01}
)

# Create your model-specific visualizer
class MyModelVisualizer(BaseWandbVisualizer):
    def log_model_specific_metrics(self, metrics):
        # Your implementation here
        pass

# Use the visualizer
visualizer = MyModelVisualizer(wandb_run=wandb_run)
visualizer.log_metrics({"accuracy": 0.95})

# Clean up
finish_wandb()
```

### Advanced Usage

```python
# Custom configuration for specific models
class PerceptronWandbVisualizer(BaseWandbVisualizer):
    def log_decision_boundary(self, X, y, model):
        # Model-specific visualization
        pass
    
    def log_learning_curve(self, losses):
        # Track training progress
        pass
```

## Architecture

The package follows these design principles:

- **Inheritance**: Use `BaseWandbVisualizer` as a parent class
- **Modularity**: Each model implements its own visualizer subclass
- **Educational Clarity**: Code prioritizes learning over performance
- **Professional Standards**: Follows industry best practices

## Development

### Dependencies

- `wandb>=0.16.0`: Experiment tracking
- `matplotlib>=3.5.0`: Visualization
- `numpy>=1.21.0`: Numerical operations

### Code Style

This package follows the project's coding guidelines:

- PEP 8 compliant formatting
- Comprehensive type hints
- Detailed docstrings (Google style)
- Educational comments and examples

## Educational Objectives

This package demonstrates:

- Professional ML experiment tracking patterns
- Clean separation of concerns in software architecture
- Consistent interfaces across different model types
- Abstract base classes and inheritance
- Dependency management and error handling

## License

Part of the AI-From-Scratch-to-Scale educational project.
