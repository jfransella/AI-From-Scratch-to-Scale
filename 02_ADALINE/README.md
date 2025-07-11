# Module 1, Model 2: ADALINE (Adaptive Linear Neuron)

## Overview

This module implements ADALINE (Adaptive Linear Neuron) from scratch, the second model in our historical journey through fundamental AI/ML algorithms. Developed by Bernard Widrow and Tedd Hoff at Stanford around 1960, ADALINE introduced the **Delta Rule** (Least Mean Squares algorithm), representing a critical evolution from the Perceptron.

## Core Innovation: The Delta Rule

ADALINE's key innovation is its learning mechanism. Unlike the Perceptron which uses binary outputs for error calculation, ADALINE uses **continuous linear outputs** to compute error:

| Model | Error Calculation Basis | Nature of Error |
| :--- | :--- | :--- |
| **Perceptron** | `predicted_binary_output` (from step function) | Binary (Correct/Incorrect) |
| **ADALINE** | `linear_output` (from summation) | Continuous (Magnitude of Error) |

### Mathematical Foundation

**Forward Pass:**
```
y = w^T * x + b
```

**Loss Function (Mean Squared Error):**
```
L = (1/2) * Σ(y_true - y_pred)^2
```

**Update Rule (Delta Rule):**
```
w_new = w_old + α * (y_true - y_pred) * x
b_new = b_old + α * (y_true - y_pred)
```

where α is the learning rate.

## Implementation Features

### 🏗️ **Architecture**
- **Single Neuron**: Linear combination of inputs with bias
- **Continuous Output**: No activation function (linear output)
- **Online Learning**: Updates weights after each sample
- **Gradient Descent**: Uses MSE loss for optimization

### 📊 **Key Components**
- `config.py`: Centralized configuration management
- `data_loader.py`: Data generation, preprocessing, and validation
- `model.py`: ADALINE implementation with mathematical rigor
- `train.py`: Comprehensive training orchestration
- `evaluate.py`: Multiple regression metrics and validation
- `visualize.py`: Educational visualizations and analysis

### 🎯 **Learning Objectives**
- Understand the mathematical foundations of gradient descent
- Implement continuous error-based learning
- Compare with Perceptron's binary learning approach
- Analyze convergence properties and stability

## Quick Start

### Environment Setup
```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1  # Windows PowerShell
source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Training ADALINE
```bash
# Run training with default synthetic data
python src/train.py

# Run with custom parameters
python src/train.py --learning_rate 0.01 --max_epochs 1000
```

### Expected Outputs
- Training progress plots showing loss convergence
- Weight evolution visualization
- Decision boundary analysis
- Model performance metrics (MSE, MAE, R²)
- Comprehensive experiment logs

## Educational Value

### Historical Context
ADALINE represents a significant step forward in neural network learning:
- **Continuous Error**: Enables gradient-based optimization
- **Stable Learning**: More reliable convergence than Perceptron
- **Mathematical Foundation**: Early form of gradient descent

### Limitations
- **Linear Separability**: Still limited to linearly separable problems
- **Single Neuron**: Cannot solve XOR or complex patterns
- **No Hidden Layers**: Lacks representational power for non-linear problems

### Transition to MLP
ADALINE's Delta Rule becomes the foundation for backpropagation in Multi-Layer Perceptrons, combining:
- Continuous error-based learning
- Multi-layer architecture
- Non-linear activation functions

## Configuration

All hyperparameters are centralized in `src/config.py`:

```python
@dataclass
class ADALINEConfig:
    # Model Architecture
    INPUT_SIZE: int = 2
    OUTPUT_SIZE: int = 1
    
    # Training Parameters
    LEARNING_RATE: float = 0.01
    MAX_EPOCHS: int = 1000
    CONVERGENCE_THRESHOLD: float = 1e-6
    
    # Data Parameters
    RANDOM_SEED: int = 42
    TRAIN_TEST_SPLIT: float = 0.8
    NORMALIZE_FEATURES: bool = True
    ADD_BIAS_TERM: bool = True
```

## Experiment Tracking

The implementation includes comprehensive experiment tracking:
- **Weights & Biases Integration**: Optional W&B logging
- **Local Logging**: Detailed training logs in `outputs/logs/`
- **Visualization**: Multiple plot types saved to `outputs/plots/`
- **Model Persistence**: Trained models saved to `outputs/`

## Performance Analysis

### Convergence Properties
- **Stability**: More stable than Perceptron due to continuous error
- **Convergence Speed**: Typically faster than Perceptron
- **Learning Rate Sensitivity**: Critical for convergence

### Comparison with Perceptron
| Aspect | Perceptron | ADALINE |
|--------|------------|---------|
| Error Type | Binary | Continuous |
| Update Rule | Perceptron Rule | Delta Rule |
| Convergence | Less Stable | More Stable |
| Learning Rate | Less Sensitive | More Sensitive |

## Detailed Documentation

For deeper analysis, see:
- **[Theoretical Deep Dive](docs/01_deep_dive.md)**: Mathematical foundations and derivations
- **[Empirical Analysis](docs/02_empirical_analysis.md)**: Experimental results and analysis

## License
This project is licensed under the MIT License.