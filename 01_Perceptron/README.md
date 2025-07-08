# Perceptron from Scratch

This project contains a complete implementation of the Perceptron algorithm from scratch using Python and NumPy. It is designed to be a clear and well-documented example of a fundamental machine learning model.

The project includes experiment tracking and hyperparameter tuning capabilities using **Weights & Biases**.

## Features

- **Pure NumPy Implementation**: The Perceptron model is built using only NumPy for core computations.
- **Experiment Tracking**: Integrated with Weights & Biases (`wandb`) to log metrics, parameters, and plots for each training run.
- **Hyperparameter Sweeps**: Includes a `sweep.yaml` configuration to easily run Bayesian hyperparameter searches.
- **Multiple Datasets**: Comes with data loaders for several classic binary classification problems.
- **Rich Visualizations**: Automatically generates and logs learning curves, confusion matrices, and decision boundaries (for 2D data).

## Available Experiments

You can run the Perceptron on any of the following pre-configured experiments:

- `and`: A simple, linearly separable logic gate.
- `xor`: A non-linearly separable logic gate (the Perceptron will fail on this).
- `mnist`: A binary classification task on handwritten digits '0' vs '1'.
- `iris-easy`: A linearly separable subset of the Iris dataset (Setosa vs. Versicolour).
- `iris-hard`: A non-linearly separable subset of the Iris dataset (Versicolour vs. Virginica).

## Getting Started

### Prerequisites

- Python 3.8+
- An account at wandb.ai (optional, for experiment tracking)

### Installation

1.  **Clone the repository** (if you haven't already).

2.  **Navigate to the project directory**:
    ```bash
    cd 01_Perceptron
    ```

3.  **Create and activate a virtual environment**:
    ```bash
    # On Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # On macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

4.  **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **(Optional) Log in to Weights & Biases**:
    If you want to track your experiments, log in to your W&B account.
    ```bash
    wandb login
    ```

## Usage

### Running a Single Experiment

You can run any of the available experiments using the `src/train.py` script. The results will be logged to your W&B project.

```bash
# Example: Run the MNIST experiment
python -m src.train --experiment mnist

# Example: Run the 'iris-hard' experiment without logging to W&B
python -m src.train --experiment iris-hard --no-wandb
```

### Running a Hyperparameter Sweep

The project is set up to run a hyperparameter sweep on the `mnist` experiment by default.

1.  **Initialize the sweep**:
    This command reads the `sweep.yaml` file and creates a new sweep on the W&B server. It will output a sweep ID.
    ```bash
    wandb sweep sweep.yaml
    ```

2.  **Run the sweep agent**:
    Copy the command from the previous step's output and run it. This will start an agent that continuously runs training jobs with different hyperparameter combinations.
    ```bash
    # Replace with the command you received
    wandb agent your-username/your-project/your-sweep-id
    ```
    You can watch the results appear in real-time on your W&B dashboard. Press `Ctrl+C` in the terminal to stop the agent.