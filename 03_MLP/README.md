# Multi-Layer Perceptron (MLP) - From Scratch Implementation

This project implements a Multi-Layer Perceptron (MLP) neural network from scratch using only NumPy, following Python best practices and coding standards.

## 🚀 Features

- **Pure NumPy Implementation**: No deep learning frameworks, built from the ground up
- **Binary & Multi-class Classification**: Support for both binary and multi-class problems
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, F1-score
- **Robustness Testing**: Built-in support for testing model robustness to data perturbations
- **Visualization Suite**: Loss curves, confusion matrices, decision boundaries, neuron weights
- **Weights & Biases Integration**: Full experiment tracking and visualization
- **Type Safety**: Complete type hints throughout the codebase
- **Error Handling**: Robust error handling and input validation
- **Reproducible Results**: Random seed control for consistent results

## 📁 Project Structure

```
03_MLP/
├── src/
│   ├── config.py          # Configuration and hyperparameters
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── model.py           # MLP model implementation
│   ├── train.py           # Training and evaluation logic
│   ├── evaluate.py        # Comprehensive evaluation utilities
│   └── visualize.py       # Plotting and visualization
├── data/                  # Dataset storage
├── notebooks/             # Jupyter notebooks for exploration  
├── outputs/               # Generated outputs (models, plots, logs)
│   ├── logs/             # Training logs
│   └── models/           # Saved model weights
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🛠️ Installation

1. **Create and activate virtual environment:**
   ```powershell
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Experiments

The project includes three pre-configured experiments:

### 1. XOR Gate (Binary Classification)
Classic non-linearly separable problem demonstrating the power of hidden layers.
```bash
python -m src.train --experiment xor
```

### 2. MNIST Multi-class Classification
Full 10-digit classification on the MNIST dataset.
```bash
python -m src.train --experiment mnist-multiclass
```

### 3. MNIST Robustness Test
Tests model robustness by evaluating on randomly shifted MNIST images.
```bash
python -m src.train --experiment mnist-failure-test
```

## 📊 Usage Examples

### Basic Training
```bash
# Train XOR classifier
python -m src.train --experiment xor

# Train MNIST classifier with W&B logging
python -m src.train --experiment mnist-multiclass

# Train without W&B logging
python -m src.train --experiment xor --no-wandb
```

### Model Evaluation
```bash
# Load and evaluate a saved model
python -m src.train --experiment mnist-multiclass --load-model outputs/models/model.npz
```

### Custom Configuration
Modify `src/config.py` to add new experiments or adjust hyperparameters:

```python
EXPERIMENTS["my_experiment"] = {
    "data_loader": my_custom_loader,
    "input_size": 784,
    "hidden_size": 64,
    "output_size": 10,
    "learning_rate": 0.01,
    "epochs": 50,
    "class_names": ["class_0", "class_1", ...],
}
```

## 🏗️ Architecture Details

### Model Features
- **Single Hidden Layer**: Configurable number of neurons
- **Activation Functions**: Sigmoid for hidden layer, Sigmoid/Softmax for output
- **Loss Functions**: MSE (binary), Cross-entropy (multi-class)
- **Optimization**: Stochastic Gradient Descent (SGD)
- **Weight Initialization**: Xavier initialization for better convergence

### Key Components

#### MLP Class (`src/model.py`)
```python
model = MLP(
    input_size=784,
    hidden_size=128,
    output_size=10,
    learning_rate=0.01,
    epochs=20,
    random_seed=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = model.score(X_test, y_test)
```

#### Evaluation Tools (`src/evaluate.py`)
```python
from src.evaluate import evaluate_model, print_evaluation_report

# Comprehensive evaluation
results = evaluate_model(model, X_test, y_test, class_names)
print_evaluation_report(results)

# Robustness testing
robustness = calculate_model_robustness(model, X_original, X_modified, y_test)
```

## 📈 Visualization

The project includes comprehensive visualization capabilities:

- **Loss Curves**: Training progress over epochs
- **Confusion Matrices**: Both counts and percentages
- **Decision Boundaries**: For 2D classification problems
- **Neuron Weights**: Visualization of learned features (MNIST)
- **Prediction Examples**: Sample predictions with confidence

## 🔧 Technical Implementation

### Type Safety & Error Handling
- Complete type hints using Python's `typing` module
- Comprehensive input validation
- Graceful error handling with informative messages
- Logging throughout the codebase

### Performance Optimizations
- NumPy vectorized operations
- Appropriate data types (float32 for memory efficiency)
- Memory cleanup for large visualizations
- Batch processing where applicable

### Reproducibility
- Random seed control (`np.random.seed(42)`)
- Deterministic model initialization
- Comprehensive logging of hyperparameters
- Model saving/loading with metadata

## 📝 Logging & Monitoring

The project uses Python's built-in logging module with multiple handlers:
- Console output for real-time monitoring  
- File logging to `outputs/logs/mlp_training.log`
- Weights & Biases integration for experiment tracking

### W&B Features
- Hyperparameter tracking
- Real-time loss monitoring
- Visualization logging
- Model artifact storage
- Experiment comparison

## 🧪 Testing & Validation

### Built-in Tests
- Input dimension validation
- Model architecture verification
- Data loading integrity checks
- Numerical stability tests

### Robustness Evaluation
The framework includes tools for testing model robustness:
- Image translation robustness (MNIST failure test)
- Performance degradation analysis
- Comparative evaluation metrics

## 📚 Dependencies

Core dependencies (see `requirements.txt` for full list):
- `numpy>=2.3.1`: Core numerical computing
- `pandas>=2.3.0`: Data manipulation  
- `matplotlib>=3.10.3`: Plotting and visualization
- `seaborn>=0.13.2`: Statistical visualizations
- `scikit-learn>=1.7.0`: Evaluation metrics
- `torch>=2.7.1`: MNIST data loading
- `wandb>=0.21.0`: Experiment tracking

## 🤝 Contributing

This implementation follows strict coding standards:
- PEP 8 compliance
- Comprehensive docstrings (Google style)
- Type hints for all functions
- Error handling and validation
- Memory-efficient operations

## 📄 License

This project is part of the AI-From-Scratch-to-Scale educational series.

## 🎓 Educational Value

This implementation demonstrates:
- **Neural Network Fundamentals**: Forward/backward propagation from scratch
- **Gradient Descent**: Manual implementation of SGD optimization
- **Software Engineering**: Best practices in ML code organization
- **Experiment Management**: Systematic approach to ML experiments
- **Visualization**: Comprehensive model interpretation techniques

Perfect for understanding the mathematical foundations of neural networks while learning professional ML development practices.
