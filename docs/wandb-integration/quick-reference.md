# W&B Integration Quick Reference

## TL;DR - Apply This Pattern to Any Model

### 1. File Structure
```
XX_ModelName/src/
├── __init__.py              # Public API
├── model.py                 # Pure algorithm (no W&B)
├── visualize.py             # Pure plotting functions
├── wandb_integration.py     # W&B experiment tracking
└── train.py                 # Orchestration
```

### 2. Create W&B Integration

```python
# wandb_integration.py
from ai_from_scratch_shared import BaseWandbVisualizer
from .visualize import plot_function1, plot_function2

class ModelNameWandbVisualizer(BaseWandbVisualizer):
    def log_training_results(self, model, X, y, predictions, **kwargs):
        # Create and log model-specific visualizations
        fig1 = plot_function1(y, predictions)
        self.log_figure(fig1, "Plot1")
        
        fig2 = plot_function2(model.training_history)
        self.log_figure(fig2, "Plot2")
```

### 3. Clean Model Implementation

```python
# model.py - NO wandb imports!
class ModelName:
    def fit(self, X, y):
        # Pure algorithm implementation
        pass
    
    def predict(self, X):
        # Pure prediction logic
        return predictions
```

### 4. Pure Visualization Functions

```python
# visualize.py
def plot_function1(y_true, y_pred):
    fig, ax = plt.subplots()
    # Create plot
    return fig

def plot_function2(training_data):
    fig, ax = plt.subplots()
    # Create plot
    return fig
```

### 5. Update Training Script

```python
# train.py
from .wandb_integration import ModelNameWandbVisualizer

def train(experiment, no_wandb=False):
    wandb.init(mode="disabled" if no_wandb else "online")
    
    model = ModelName()
    model.fit(X, y)
    predictions = model.predict(X)
    
    if not no_wandb:
        visualizer = ModelNameWandbVisualizer()
        visualizer.log_training_results(model, X, y, predictions)
    
    wandb.finish()
```

### 6. Define Public API

```python
# __init__.py
from .model import ModelName
from .wandb_integration import ModelNameWandbVisualizer
from .visualize import plot_function1, plot_function2

__all__ = ['ModelName', 'ModelNameWandbVisualizer', 
           'plot_function1', 'plot_function2']
```

## Success Checklist

- [ ] Model has NO wandb imports
- [ ] Visualize functions return matplotlib figures  
- [ ] W&B integration extends BaseWandbVisualizer
- [ ] Train script uses visualizer conditionally
- [ ] Package has proper `__init__.py`
- [ ] All imports work: `from src import ModelName, ModelNameWandbVisualizer`

## Working Examples

✅ **01_Perceptron** - Binary classification with decision boundaries  
✅ **04_Hopfield_Network** - Associative memory with pattern visualization  
✅ **03_MLP** - Multi-layer networks with weight visualization

See `WANDB_STANDARDIZATION_GUIDE.md` for complete documentation.
