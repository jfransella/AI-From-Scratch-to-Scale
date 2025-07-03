# -*- coding: utf-8 -*-
"""Main training script for the Perceptron model.

This script orchestrates the training process for different experiments.
It uses command-line arguments to select the experiment to run.

Supported experiments:
- 'logic_gate': Trains the Perceptron on a simple, 2D logic gate dataset.
- 'mnist': Trains the Perceptron on a filtered MNIST dataset (digits 0 and 1).

Example usage:
    python src/train.py --experiment logic_gate
    python src/train.py --experiment mnist
"""

import logging
import os
import argparse
import numpy as np  

from src import config
# Import both data loaders
from src.data_loader import load_perceptron_data, load_mnist_data
from src.model import Perceptron
# Import the visualization function
from src.visualize import plot_decision_boundary, plot_perceptron_weights, plot_learning_curve


# --- Logging Setup ---
# This setup is unchanged
os.makedirs("outputs/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/training.log'),
        logging.StreamHandler()
    ]
)


def train(experiment):
    """Main function to run a selected training experiment."""
    logging.info(f"--- Starting Perceptron Training: '{experiment}' experiment ---")

    # --- 1. Load Data & Set Parameters based on experiment ---
    if experiment == 'logic_gate':
        logging.info(f"Loading data from: {config.INPUT_DATA_PATH}")
        try:
            X, y = load_perceptron_data(config.INPUT_DATA_PATH)
            lr = config.LOGIC_GATE_LEARNING_RATE
            epochs = config.LOGIC_GATE_EPOCHS
            visualize = True
        except FileNotFoundError:
            logging.error(f"Error: Data file not found at {config.INPUT_DATA_PATH}.")
            return
    elif experiment == 'mnist':
        logging.info("Loading MNIST data for digits 0 and 1...")
        X, y = load_mnist_data()
        lr = config.MNIST_LEARNING_RATE
        epochs = config.MNIST_EPOCHS
        visualize = False  # 2D visualization is not applicable for 784-D data
    else:
        logging.error(f"Unknown experiment: {experiment}")
        return

    logging.info(f"Data loaded successfully. Found {X.shape[0]} samples with {X.shape[1]} features.")

    # --- 2. Initialize and train the model ---
    logging.info(f"Initializing Perceptron with LR={lr} and Epochs={epochs}.")
    perceptron = Perceptron(learning_rate=lr, n_iters=epochs, logger=logging)

    logging.info("Training started...")
    perceptron.fit(X, y)
    logging.info("Training complete.")

    # --- 3. Evaluate and Visualize ---
    predictions = perceptron.predict(X)
    # Ensure y is binary for accuracy calculation
    y_binary = np.array([1 if i > 0 else 0 for i in y])
    accuracy = (predictions == y_binary).mean()
    logging.info(f"Final training accuracy: {accuracy:.4f}")

    # --- 4. Visualize ---
    # Generate a learning curve for both experiments
    logging.info("Generating learning curve plot...")
    learning_curve_filename = f"learning_curve_{experiment}.png"
    plot_learning_curve(perceptron.errors_per_epoch, filename=learning_curve_filename)

    # Generate experiment-specific visualizations
    if experiment == 'logic_gate':
        logging.info("Generating decision boundary plot...")
        plot_filename = f"decision_boundary_{config.INPUT_DATA_PATH.split('/')[-1].split('.')[0]}.png"
        plot_decision_boundary(X, y, perceptron, filename=plot_filename)
    elif experiment == 'mnist':
        logging.info("Generating model weights visualization...")
        plot_perceptron_weights(perceptron)

        logging.info(f"--- Experiment '{experiment}' Finished ---")


if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Run a Perceptron training experiment.")
    parser.add_argument(
        '--experiment',
        type=str,
        default='logic_gate',
        choices=['logic_gate', 'mnist'],
        help="The experiment to run ('logic_gate' or 'mnist')."
    )
    args = parser.parse_args()
    train(args.experiment)