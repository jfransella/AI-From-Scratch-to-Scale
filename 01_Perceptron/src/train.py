# -*- coding: utf-8 -*-
"""Main training script for the Perceptron model.

This script orchestrates the training process for different experiments.
It uses command-line arguments to select the experiment to run.

Example usage:
    python src/train.py --experiment and
    python src/train.py --experiment xor
    python src/train.py --experiment mnist
"""

import logging
import os
import argparse
import numpy as np

from src import config
from src.data_loader import load_perceptron_data, load_mnist_data
from src.model import Perceptron
from src.visualize import plot_decision_boundary, plot_perceptron_weights, plot_learning_curve

# --- Logging Setup ---
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
    if experiment in config.LOGIC_GATE_DATA_PATHS:
        data_path = config.LOGIC_GATE_DATA_PATHS[experiment]
        logging.info(f"Loading data from: {data_path}")
        try:
            X, y = load_perceptron_data(data_path)
            lr = config.LOGIC_GATE_LEARNING_RATE
            epochs = config.LOGIC_GATE_EPOCHS
        except FileNotFoundError:
            logging.error(f"Error: Data file not found at {data_path}.")
            return
    elif experiment == 'mnist':
        logging.info("Loading MNIST data for digits 0 and 1...")
        X, y = load_mnist_data()
        lr = config.MNIST_LEARNING_RATE
        epochs = config.MNIST_EPOCHS
    else:
        logging.error(f"Unknown experiment: {experiment}")
        return

    logging.info(f"Data loaded successfully. Found {X.shape[0]} samples with {X.shape[1]} features.")

    # --- 2. Initialize and train the model ---
    logging.info(f"Initializing Perceptron with LR={lr} and Epochs={epochs}.")
    perceptron = Perceptron(learning_rate=lr, n_iters=epochs, logger=logging)
    perceptron.fit(X, y)
    logging.info("Training complete.")

    # --- 3. Evaluate ---
    predictions = perceptron.predict(X)
    y_binary = np.array([1 if i > 0 else 0 for i in y])
    accuracy = (predictions == y_binary).mean()
    logging.info(f"Final training accuracy: {accuracy:.4f}")

    # --- 4. Visualize ---
    logging.info("Generating learning curve plot...")
    plot_learning_curve(perceptron.errors_per_epoch, filename=f"learning_curve_{experiment}.png")

    if experiment in config.LOGIC_GATE_DATA_PATHS:
        logging.info("Generating decision boundary plot...")
        plot_decision_boundary(X, y, perceptron, filename=f"decision_boundary_{experiment}.png")
    elif experiment == 'mnist':
        logging.info("Generating model weights visualization...")
        plot_perceptron_weights(perceptron, filename=f"perceptron_weights_{experiment}.png")

    logging.info(f"--- Experiment '{experiment}' Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Perceptron training experiment.")
    parser.add_argument(
        '--experiment',
        type=str,
        default='and',
        choices=['and', 'xor', 'mnist'],
        help="The experiment to run ('and', 'xor', or 'mnist')."
    )
    args = parser.parse_args()
    train(args.experiment)