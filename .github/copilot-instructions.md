# Python Coding Guidelines for AI-From-Scratch-to-Scale

## Code Style & Formatting
- Follow PEP 8 style guidelines
- Use 4 spaces for indentation (no tabs)
- Maximum line length of 88 characters (Black formatter standard)
- Use descriptive variable names: `learning_rate` not `lr`, `num_epochs` not `n_epochs`
- Use snake_case for functions and variables, PascalCase for classes
- Add blank lines around class and function definitions

## Type Hints & Documentation
- Always use type hints for function parameters and return values
- Use docstrings for all classes and functions (Google or NumPy style)
- Include parameter descriptions, return value descriptions, and examples where helpful
- Use `typing` module for complex types: `List[float]`, `Optional[int]`, `Tuple[np.ndarray, np.ndarray]`

```python
def train_model(X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01) -> Dict[str, float]:
    """Train the model on the given data.
    
    Args:
        X: Input features of shape (n_samples, n_features)
        y: Target labels of shape (n_samples,)
        learning_rate: Step size for gradient descent
        
    Returns:
        Dictionary containing training metrics
    """
```

## Project Structure Standards
- Each model implementation should follow this structure:
  ```
  XX_ModelName/
  ├── src/
  │   ├── config.py          # Configuration and hyperparameters
  │   ├── data_loader.py     # Data loading and preprocessing
  │   ├── model.py           # Model implementation
  │   ├── train.py           # Training logic
  │   ├── evaluate.py        # Evaluation and testing
  │   └── visualize.py       # Plotting and visualization
  ├── data/                  # Dataset storage
  ├── notebooks/             # Jupyter notebooks for exploration  
  ├── outputs/               # Generated outputs (models, plots, logs)
  ├── requirements.txt       # Dependencies
  └── README.md             # Model-specific documentation
  ```

## Virtual Environment Setup
- Each project folder should have its own virtual environment in `.venv/`
- Standard activation commands:
  ```bash
  # Windows (PowerShell)
  .venv\Scripts\Activate.ps1
  
  # Windows (Command Prompt)
  .venv\Scripts\activate.bat
  
  # macOS/Linux
  source .venv/bin/activate
  ```
- Cross-platform shell script provided: `activate-env.sh`

## Machine Learning Best Practices
- Always set random seeds for reproducibility: `np.random.seed(42)`, `torch.manual_seed(42)`
- Separate training, validation, and test sets
- Log important metrics and hyperparameters
- Save model checkpoints during training
- Use configuration files for hyperparameters instead of hardcoding
- Implement proper error handling for data loading and model operations

## NumPy & Scientific Computing
- Use vectorized operations instead of loops when possible
- Prefer `np.array` operations over list comprehensions for numerical data
- Use broadcasting effectively for element-wise operations
- Always specify data types: `np.zeros(shape, dtype=np.float32)`
- Use `np.random.Generator` for modern random number generation

## Error Handling & Validation
- Validate input shapes and types at function entry points
- Use assertions for debugging: `assert X.shape[1] == self.input_size, "Input dimension mismatch"`
- Handle common ML errors gracefully (convergence issues, numerical instability)
- Provide meaningful error messages that help debugging

## Dependencies & Environment
- Pin dependency versions in requirements.txt for reproducibility
- Use virtual environments for each model implementation
- Import only what you need: `from sklearn.metrics import accuracy_score` not `from sklearn import *`
- Group imports: standard library, third-party, local modules

## Performance & Memory
- Use appropriate data types (float32 vs float64) based on precision needs
- Free up memory when working with large datasets: `del large_array`
- Profile code for bottlenecks in computationally intensive sections
- Use generators for large datasets that don't fit in memory

## Testing & Validation
- Include basic unit tests for core model functions
- Test with toy datasets to verify correct implementation
- Compare outputs with known implementations when possible
- Test edge cases (empty data, single sample, etc.)

## Logging & Monitoring
- Use Python's `logging` module instead of print statements
- Log training progress, loss values, and important metrics
- Save training logs to files for later analysis
- Use meaningful log levels (DEBUG, INFO, WARNING, ERROR)

## Example Function Template
```python
import logging
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

def forward_pass(X: np.ndarray, weights: np.ndarray, bias: float) -> Tuple[np.ndarray, np.ndarray]:
    """Perform forward pass through the network.
    
    Args:
        X: Input data of shape (batch_size, input_dim)
        weights: Weight matrix of shape (input_dim, output_dim)
        bias: Bias term
        
    Returns:
        Tuple of (activations, pre_activations)
        
    Raises:
        ValueError: If input dimensions don't match weight dimensions
    """
    if X.shape[1] != weights.shape[0]:
        raise ValueError(f"Input dim {X.shape[1]} doesn't match weight dim {weights.shape[0]}")
    
    logger.debug(f"Forward pass with input shape: {X.shape}")
    
    pre_activations = X @ weights + bias
    activations = np.tanh(pre_activations)  # Example activation
    
    return activations, pre_activations
```

## Model Implementation Guidelines
- Keep model classes focused and single-purpose
- Implement `fit()`, `predict()`, and `score()` methods for consistency
- Use separate methods for forward pass, backward pass, and parameter updates
- Store hyperparameters as class attributes
- Implement proper `__repr__()` methods for debugging

## Visualization Standards
- Use consistent color schemes and styling across all plots
- Include proper labels, titles, and legends
- Save plots in high resolution (300 DPI) for documentation
- Use meaningful figure sizes: `plt.figure(figsize=(10, 6))`
- Close figure objects to prevent memory leaks: `plt.close()`