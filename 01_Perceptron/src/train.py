# -*- coding: utf-8 -*-
"""Main training script for the Perceptron model.

This script orchestrates the training process by:
1. Loading configuration settings from the config module.
2. Setting up standardized logging for console and file output.
3. Loading the dataset using the data_loader module.
4. Instantiating and training the Perceptron model.
5. Evaluating the trained model's accuracy on the training data.

"""

import logging
import os

from src import config
from src.data_loader import load_perceptron_data
from src.model import Perceptron

# --- Logging Setup ---
# Create the outputs directory if it doesn't exist
os.makedirs("outputs/logs", exist_ok=True)

# Configure console logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Configure file logger
file_handler = logging.FileHandler('outputs/logs/training.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def train():
    """Main function to run the Perceptron training process."""
    logger.info("--- Starting Perceptron Training ---")

    # 1. Load data
    logger.info(f"Loading data from: {config.INPUT_DATA_PATH}")
    try:
        X, y = load_perceptron_data(config.INPUT_DATA_PATH)
        logger.info(f"Data loaded successfully. Found {X.shape[0]} samples with {X.shape[1]} features.")
    except FileNotFoundError:
        logger.error(f"Error: Data file not found at {config.INPUT_DATA_PATH}. Please ensure the data exists.")
        return

    # 2. Initialize and train the model
    logger.info(f"Initializing Perceptron with LR={config.LEARNING_RATE} and Epochs={config.EPOCHS}.")
    perceptron = Perceptron(learning_rate=config.LEARNING_RATE, n_iters=config.EPOCHS)

    logger.info("Training started...")
    perceptron.fit(X, y)
    logger.info("Training complete.")

    # 3. Evaluate on training data to check for convergence
    predictions = perceptron.predict(X)
    accuracy = (predictions == y).mean()
    logger.info(f"Final training accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    train()