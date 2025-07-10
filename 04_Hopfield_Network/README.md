# Hopfield Network: Energy-Based Associative Memory

## Overview

This module implements the classic Hopfield Network (1982), demonstrating energy-based learning as an alternative paradigm to gradient-based methods. The implementation prioritizes educational clarity to help understand energy functions, Lyapunov stability, and associative memory concepts.

## Educational Objectives

- **Energy-Based Learning**: Understand how energy functions enable pattern storage and retrieval
- **Lyapunov Stability**: Learn how energy minimization guarantees convergence
- **Associative Memory**: Experience content-addressable memory vs. supervised classification
- **Statistical Mechanics**: Explore connections to physics (Ising model, spin glasses)
- **Historical Context**: Appreciate Hopfield's bridge between neuroscience and physics

## Mathematical Foundation

### Energy Function
```
E = -0.5 * Σ(i,j) w_ij * s_i * s_j
```

### Hebbian Storage Rule
```
w_ij = (1/N) * Σ(μ) ξ_i^μ * ξ_j^μ
```

### Update Rule
```
s_i = sign(Σ(j) w_ij * s_j)
```

## Quick Start

### Installation
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```bash
# Run basic pattern training experiment
python src/train.py --experiment basic --no-wandb

# Run complete experimental suite
python src/train.py --experiment all --no-wandb

# With Weights & Biases tracking
python src/train.py --experiment all
```

## Experiments

### 1. Basic Pattern Training
Demonstrates one-shot Hebbian learning with simple geometric patterns.

### 2. Storage Capacity Analysis
Tests the theoretical capacity limit (~0.15 * N patterns) empirically.

### 3. Noise Robustness
Shows error correction capabilities with corrupted input patterns.

### 4. Convergence Dynamics
Analyzes energy minimization and Lyapunov function properties.

## Project Structure

```
04_Hopfield_Network/
├── src/
│   ├── config.py              # All hyperparameters and constants
│   ├── model.py               # Core Hopfield Network implementation
│   ├── data_loader.py         # Pattern generation and loading
│   ├── train.py               # Training experiments and analysis
│   ├── wandb_integration.py   # Experiment tracking integration
│   └── visualize.py           # Visualization utilities
├── data/                      # Generated datasets
├── outputs/                   # Experimental results
│   ├── models/                # Trained network weights
│   ├── plots/                 # Generated visualizations
│   └── logs/                  # Training logs
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Key Features

### Educational Focus
- Extensive mathematical comments explaining energy-based learning
- Visualization of energy landscapes and convergence dynamics
- Comparison with gradient-based supervised learning paradigms
- Historical context and connections to statistical mechanics

### Professional Implementation
- Type hints and comprehensive docstrings
- Configurable parameters in `config.py`
- Dual logging (console + file)
- Optional Weights & Biases integration
- Reproducible experiments with fixed random seeds

### Experimental Suite
- Pattern storage capacity analysis
- Noise robustness testing
- Energy convergence visualization
- Comprehensive reporting and artifact saving

## Mathematical Insights

### Energy-Based vs. Gradient-Based Learning
- **Hopfield**: Energy minimization, one-shot storage, no error signal
- **Backpropagation**: Gradient descent, iterative optimization, supervised error

### Capacity Limitations
- Theoretical limit: ~0.15 * N patterns for reliable retrieval
- Spurious states emerge beyond capacity
- Pattern interference increases with storage load

### Biological Plausibility
- Local learning rules (Hebbian)
- Symmetric connectivity
- Binary neuron states
- No external teacher signal

## Historical Context

John Hopfield (1982) revolutionized neural network research by:
- Connecting neural computation to statistical mechanics
- Showing guaranteed convergence via energy functions
- Demonstrating content-addressable memory
- Inspiring spin glass theory applications

## References

1. Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. PNAS.
2. Amit, D. J. (1989). Modeling Brain Function: The World of Attractor Neural Networks.
3. Hertz, J., Krogh, A., & Palmer, R. G. (1991). Introduction to the Theory of Neural Computation.

## Learning Outcomes

After working through this implementation, you should understand:
- How energy functions enable unsupervised pattern storage
- Why symmetric weights guarantee convergence
- The relationship between network capacity and pattern interference
- Connections between neural computation and statistical mechanics
- Differences between energy-based and gradient-based learning paradigms
