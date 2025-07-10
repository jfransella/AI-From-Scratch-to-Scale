# Python Coding Guidelines for AI-From-Scratch-to-Scale

## Project Context & Learning Objectives

This repository implements fundamental AI/ML algorithms from scratch to provide deep understanding of core concepts. The project follows a progressive learning path from basic perceptrons to advanced transformers and generative models.

### Educational Philosophy
- **Implementation First**: Build algorithms from scratch using only NumPy/basic libraries before using frameworks
- **Mathematical Understanding**: Each implementation includes detailed mathematical explanations and derivations
- **Practical Application**: Every model includes real datasets and practical examples
- **Code Quality**: Professional-grade code with proper documentation, testing, and error handling
- **Reproducibility**: All experiments are reproducible with fixed random seeds and documented environments

### Learning Progression
1. **Foundations (01-04)**: Basic neural units, linear models, simple networks
2. **Deep Learning (05-14)**: CNNs, object detection, segmentation architectures  
3. **Sequential Models (15-18)**: RNNs, LSTMs, attention mechanisms, transformers
4. **Generative Models (19-22)**: VAEs, GANs, diffusion models
5. **Advanced Topics (23-25)**: Graph networks, modern language models, efficient architectures

### Key Learning Outcomes
- Understand the mathematical foundations of each algorithm
- Implement complex models using only basic libraries
- Apply proper software engineering practices to ML code
- Develop intuition for hyperparameter tuning and model debugging
- Create reproducible and well-documented ML experiments

### Code Generation Context
When generating code for this project:
- Prioritize educational clarity over performance optimization
- Include extensive comments explaining mathematical operations
- Add assertions to validate intermediate computations
- Implement comprehensive logging for training dynamics
- Create modular, testable components that can be easily understood
- Include practical examples and real-world applicationsz
- Implement comprehensive visualizations
- Leverage Weights & Biases for experiment & visualization tracking

## Code Style & Formatting
- Follow PEP 8 style guidelines
- Use 4 spaces for indentation (no tabs)
- Maximum line length of 88 characters (Black formatter standard)
- Use descriptive variable names: `learning_rate` not `lr`, `num_epochs` not `n_epochs`
- Use snake_case for functions and variables, PascalCase for classes
- Add blank lines around class and function definitions
- **Auto-formatting**: Use Black code formatter for consistent formatting
- **Linting**: Use flake8 to catch common errors and style issues

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
- Group imports in standard order: 1) standard library, 2) third-party libraries, 3) local modules
- Sort imports alphabetically within each group
- Use absolute imports for clarity

## Performance & Memory
- Use appropriate data types (float32 vs float64) based on precision needs
- Free up memory when working with large datasets: `del large_array`
- Profile code for bottlenecks in computationally intensive sections
- Use generators for large datasets that don't fit in memory

### Performance Strategy by Model Complexity
- **Early Models (NumPy-based)**: Focus on algorithmic clarity over optimization
  - Use NumPy vectorization over Python loops
  - Avoid GPU acceleration to understand computational constraints historically
  - Prioritize educational value over execution speed
- **Later Models (Framework-based)**: Apply modern optimization practices
  - Implement device-agnostic code: automatic GPU/CPU detection
  - Use framework DataLoaders with multi-processing for larger datasets
  - Apply profiling tools (e.g., torch.profiler) when performance issues arise

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

### Dual Logging Strategy
- **Console Output**: Simple, readable status updates for real-time monitoring
- **File Logging**: Detailed, machine-parsable logs saved to `outputs/logs/training.log`
- **Saved Artifacts**: All outputs (visualizations, logs, model weights) saved to appropriate `outputs/` subdirectories

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

## Configuration Management
- **No Hardcoded Values**: Hyperparameters (learning rate, batch size, epochs), file paths, and settings must not be hardcoded in training or model scripts
- **Centralized Config**: Store all configuration values in `src/config.py` for easy experimentation
- **Constants Naming**: Use ALL_CAPS for constants (e.g., `LEARNING_RATE`, `BATCH_SIZE`)

## Modularity & Single Responsibility
- **Single Purpose**: Each script should have one clear responsibility:
  - `data_loader.py`: Only data loading and preprocessing
  - `model.py`: Only network architecture definition
  - `train.py`: Orchestrates training by importing from other modules
- **Framework Best Practices**: When using PyTorch/TensorFlow:
  - Define models as subclasses of `torch.nn.Module`
  - Explicitly manage device placement with `.to(device)`
  - Use framework-specific best practices without obscuring core concepts

## Development Workflow Guidelines
- **File Development Order**: config.py → data_loader.py → model.py → train.py → evaluate.py → visualize.py
- **Atomic Commits**: Commit frequently with small, logical changes
- **Quality Review Checklist**:
  - Correctness: Meets requirements and learning objectives
  - Clarity: Follows "Code as a Learning Tool" philosophy  
  - Compliance: Adheres to coding standards (naming, docstrings, type hints)
  - Educational Value: Easy to understand and learn from

## Git & Version Control
- **Branching**: Use feature branches for each model (feature/XX_ModelName)
- **Commit Messages**: Follow Conventional Commits specification (feat, fix, docs, test, refactor, chore)
- **Pull Requests**: Required for all changes to main branch
- **Documentation**: Each model requires comprehensive README following project template

## GitHub Copilot MCP Integration

### Issue-Driven Development
- **Automatic Issue Creation**: When starting a new model, create GitHub issues to track progress
- **Issue Templates**: Use standardized issue templates for model implementation milestones
- **Progress Tracking**: Link commits and PRs to issues using GitHub keywords (fixes #123, closes #456)
- **Educational Documentation**: Create issues for mathematical concepts that need explanation

### Enhanced Pull Request Workflow
- **Automated PR Creation**: Use MCP to create PRs directly from VS Code when feature branches are ready
- **Code Review Integration**: Request Copilot code reviews for educational feedback on implementations
- **Draft PRs**: Create draft PRs early for work-in-progress visibility and collaboration
- **Educational Context**: Include mathematical explanations and learning objectives in PR descriptions

### Project Management Integration
- **Milestone Tracking**: Create GitHub milestones for each learning module (01-04 Foundations, 05-14 Deep Learning, etc.)
- **Assignment Workflows**: Assign Copilot to specific implementation issues for automated assistance
- **Progress Visualization**: Use GitHub Projects to track implementation status across all 25 models
- **Learning Path Management**: Organize issues by educational priority and dependencies

### Repository Organization
- **Branch Protection**: Use MCP to configure branch protection rules for main branch
- **Release Management**: Create releases for completed learning modules with comprehensive documentation
- **Wiki Integration**: Maintain educational content and mathematical derivations in GitHub Wiki
- **Notification Management**: Configure GitHub notifications to stay updated on educational discussions

### Code Quality Automation
- **Automated Testing**: Link GitHub Actions workflows to MCP for continuous integration
- **Code Review Requests**: Automatically request Copilot reviews for educational code quality
- **Documentation Updates**: Use MCP to update documentation when implementations change
- **Learning Outcome Validation**: Create issues to verify educational objectives are met

### Educational Collaboration
- **Community Engagement**: Use GitHub Discussions for mathematical questions and learning insights
- **Student Progress**: Track learning progress through GitHub issue completion
- **Knowledge Sharing**: Create public issues for common implementation challenges and solutions
- **Historical Context**: Document the evolution of neural network concepts through commit history