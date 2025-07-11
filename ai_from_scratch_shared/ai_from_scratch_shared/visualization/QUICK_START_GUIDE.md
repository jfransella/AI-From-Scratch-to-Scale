# Shared Visualization Framework - Quick Start Guide

## Overview
The shared visualization framework provides consistent, educational visualizations across all AI models in the project. All models now use the same styling, annotations, and best practices.

## Basic Usage

### Importing the Framework
```python
from ai_from_scratch_shared.visualization import (
    BaseVisualizer,
    ConfusionMatrixVisualizer,
    TrainingCurveVisualizer,
    DecisionBoundaryVisualizer,
    DataDistributionVisualizer
)
```

### Creating a Model-Specific Visualizer
```python
from ai_from_scratch_shared.visualization import BaseVisualizer
from pathlib import Path

class MyModelVisualizer(BaseVisualizer):
    """Visualizer for MyModel using shared framework."""
    
    def __init__(self, default_save_dir: Path = None):
        super().__init__(default_save_dir)
    
    def visualize_my_specific_plot(self, data, title="My Plot", show=True):
        """Create model-specific visualization."""
        fig, ax = self.create_figure(figsize=(10, 6))
        
        # Your plotting code here
        ax.plot(data)
        ax.set_title(title)
        
        # Add educational context
        self.add_educational_annotation(
            ax, 
            "This plot shows the model's learning progress",
            "Educational insight: Notice how the curve converges"
        )
        
        return self.save_and_show(fig, "my_plot.png", show)
```

### Using Common Visualizers
```python
# Confusion Matrix
cm_viz = ConfusionMatrixVisualizer()
fig, ax = cm_viz.plot(y_true, y_pred, class_names, show=True)

# Training Curves
tc_viz = TrainingCurveVisualizer()
fig, axes = tc_viz.plot(train_loss, val_loss, train_acc, val_acc, show=True)

# Decision Boundary
db_viz = DecisionBoundaryVisualizer()
fig, ax = db_viz.plot(X, y, model, show=True)

# Data Distribution
dd_viz = DataDistributionVisualizer()
fig, axes = dd_viz.plot(data, labels, show=True)
```

## Educational Features

### Automatic Annotations
All visualizations include:
- Mathematical context relevant to the algorithm
- Performance insights and interpretations
- Learning objective connections
- Professional styling optimized for education

### Consistent Styling
- Educational color palettes
- Professional typography
- Standardized figure sizes
- Academic presentation quality

## Model Integration Examples

### Hopfield Network (Completed)
```python
from src.visualize_new import HopfieldVisualizer

viz = HopfieldVisualizer(default_save_dir=Path("outputs/plots"))

# Pattern visualization
fig, ax = viz.visualize_pattern(pattern, "Pattern A", show=True)

# Energy landscape
fig, ax = viz.visualize_energy_landscape(patterns, weights, show=True)

# Convergence analysis
fig, axes = viz.plot_convergence_analysis(convergence_data, show=True)
```

### Backwards Compatibility
All existing visualization functions continue to work:
```python
# Old way - still works
from src.visualize import visualize_pattern
visualize_pattern(pattern, "Pattern A")

# New way - enhanced with shared framework
from src.visualize_new import HopfieldVisualizer
viz = HopfieldVisualizer()
viz.visualize_pattern(pattern, "Pattern A")
```

## Migration Strategy

### For Existing Models
1. Create new `visualize_new.py` file with shared framework integration
2. Keep original `visualize.py` for backwards compatibility
3. Update training scripts to use new visualizer
4. Gradually migrate all visualization calls

### For New Models
1. Create model-specific visualizer class extending `BaseVisualizer`
2. Implement required visualization methods
3. Use common visualizers where applicable
4. Add educational annotations for learning context

## Best Practices

### File Organization
```
model_directory/
├── src/
│   ├── visualize.py          # Original (keep for compatibility)
│   ├── visualize_new.py      # New shared framework integration
│   └── train.py              # Updated to use shared framework
```

### Code Structure
```python
class ModelVisualizer(BaseVisualizer):
    """Model-specific visualizer using shared framework."""
    
    def __init__(self, default_save_dir: Path = None):
        super().__init__(default_save_dir)
        # Model-specific initialization
    
    def visualize_model_specific(self, data, **kwargs):
        """Model-specific visualization method."""
        fig, ax = self.create_figure()
        # Plotting logic
        self.add_educational_annotation(ax, context, insight)
        return self.save_and_show(fig, filename, show)

# Backwards compatibility functions
def legacy_function(data, **kwargs):
    """Wrapper for backwards compatibility."""
    viz = ModelVisualizer()
    return viz.visualize_model_specific(data, **kwargs)
```

### Educational Annotations
```python
# Add context and insights to every plot
self.add_educational_annotation(
    ax,
    mathematical_context="The loss function L(θ) = ...",
    performance_insight="Notice the convergence after epoch 50",
    concept_connection="This demonstrates gradient descent optimization"
)
```

## Framework Components

### Core Classes
- **`BaseVisualizer`**: Foundation class for all visualizers
- **`EducationalAnnotator`**: Adds learning context to plots
- **Common Visualizers**: Ready-to-use components for standard ML plots

### Styling System
- **Educational themes**: Optimized for learning and presentations
- **Color schemes**: Consistent, accessible, professional
- **Typography**: Clear, readable fonts and sizes

### Utilities
- **Plot management**: Automatic saving, cleanup, memory management
- **File handling**: Standardized paths and naming
- **Error handling**: Graceful failure and informative messages

## Examples by Model Type

### Neural Networks (Perceptron, MLP)
- Training curves with loss/accuracy
- Decision boundaries for classification
- Weight visualizations
- Activation function plots

### Convolutional Networks (LeNet, AlexNet, VGG, ResNet)
- Feature map visualizations
- Filter/kernel displays
- Training progression plots
- Architecture diagrams

### Recurrent Networks (RNN, LSTM, GRU)
- Sequence prediction plots
- Hidden state visualizations
- Attention weight heatmaps
- Temporal pattern analysis

### Generative Models (VAE, GAN, DDPM)
- Generated sample galleries
- Latent space visualizations
- Training dynamics (discriminator vs generator)
- Quality metric plots

### Graph Networks (GCN)
- Node/edge visualizations
- Graph structure plots
- Embedding space displays
- Message passing illustrations

## Support and Documentation

### Getting Help
- Check existing model integrations for examples
- Review shared framework API documentation
- Use consistent patterns across models
- Add educational value to every visualization

### Contributing
- Follow established patterns for new visualizers
- Add comprehensive docstrings and type hints
- Include educational annotations in all plots
- Test backwards compatibility thoroughly

---
*Shared Visualization Framework - AI From Scratch to Scale Project*
