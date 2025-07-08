# -*- coding: utf-8 -*-
"""Main training and evaluation script for the Multi-Layer Perceptron (MLP) model.

This script serves as the main entry point for running experiments. It handles:
- Data loading based on the selected experiment from `config.py`.
- Model initialization and training using backpropagation.
- Integration with Weights & Biases (wandb) for logging metrics, parameters,
  and visualizations.
- Support for hyperparameter sweeps using wandb agents.

Example usage:
    # Run the XOR experiment
    python -m src.train --experiment xor
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import wandb

from src.config import EXPERIMENTS, WANDB_PROJECT_NAME
from src.model import MLP
from src.visualize import Visualizer


def train(experiment, no_wandb=False, load_model_path=None):
    """Orchestrates a single training and evaluation run for a given experiment."""
    logging.info(f"--- Starting MLP Training: '{experiment}' experiment ---")

    # --- 1. Prepare Experiment ---
    if experiment not in EXPERIMENTS:
        logging.error(f"Unknown experiment: {experiment}. Check config.py.")
        return

    exp_config = EXPERIMENTS[experiment]
    logging.info(f"Loading data for '{experiment}' experiment...")
    data_tuple = exp_config["data_loader"]()
    if len(data_tuple) == 4:
        X_train, y_train, X_test, y_test = data_tuple
        logging.info(f"Data loaded successfully. Found {X_train.shape[0]} training samples and {X_test.shape[0]} test samples.")
    else:
        X_train, y_train = data_tuple
        X_test, y_test = X_train, y_train  # Fallback for old data loaders
        logging.info(f"Data loaded successfully. Found {X_train.shape[0]} samples.")


    # --- Setup W&B ---
    wandb_mode = "disabled" if no_wandb else "online"
    wandb.init(
        mode=wandb_mode,
        project=WANDB_PROJECT_NAME,
        config={
            "learning_rate": exp_config["learning_rate"],
            "epochs": exp_config["epochs"],
            "hidden_size": exp_config["hidden_size"],
            "experiment_type": experiment,
        }
    )
    if no_wandb:
        logging.info("Weights & Biases logging is disabled for this run.")
    else:
        logging.info(f"Weights & Biases run '{wandb.run.name}' initialized.")

    visualizer = Visualizer(wandb, enabled=(not no_wandb))

    # Get hyperparameters from the W&B config (allows for sweeps)
    lr = wandb.config.learning_rate
    epochs = wandb.config.epochs
    hidden_size = wandb.config.hidden_size

    model = MLP(
        input_size=exp_config["input_size"],
        hidden_size=hidden_size,
        output_size=exp_config["output_size"],
        learning_rate=lr,
        epochs=epochs,
        logger=logging,
        wandb_run=wandb,
    )

    # --- 2. Load or Train Model ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if load_model_path:
        logging.info(f"Loading pre-trained model from {load_model_path}")
        model.load_model(load_model_path)
        # Use the loaded model's name for the run, with an 'eval-' prefix
        base_name = os.path.basename(load_model_path).replace('.npz', '')
        run_name = f"eval-{base_name}-{timestamp}"
    else:
        # Set a descriptive run name for a new training run
        run_name = f"{experiment}-{timestamp}-lr_{lr:.4f}-hs_{hidden_size}-ep_{epochs}"
        logging.info(f"Initializing MLP for new training run: {run_name}")
        model.fit(X_train, y_train)
        logging.info("Training complete.")

        # Save the newly trained model
        model_dir = "outputs/models"
        model_path = os.path.join(model_dir, f"{run_name}.npz")
        model.save_model(model_path)

    if not no_wandb:
        wandb.run.name = run_name
        logging.info(f"W&B run name updated to: {run_name}")


    # --- 3. Evaluate ---
    predictions = model.predict(X_test)
    # For multi-class, y is one-hot encoded. We need to get the class index.
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_labels = np.argmax(y_test, axis=1)
    else: # For binary, y is already a column vector of labels
        y_labels = y_test.flatten()
    accuracy = (predictions == y_labels).mean()
    logging.info(f"Final evaluation accuracy: {accuracy:.4f}")
    wandb.summary["final_eval_accuracy"] = accuracy

    # --- 4. Visualize ---
    class_names = exp_config.get("class_names")
    visualizer.log_all(
        model=model,
        X=X_test,
        y=y_test,
        predictions=predictions,
        class_names=class_names
    )

    wandb.finish()
    logging.info(f"--- Experiment '{experiment}' Finished ---")


def main():
    """Sets up logging, parses arguments, and starts the training process."""
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outputs/logs/mlp_training.log'),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description="Run an MLP training experiment.")
    parser.add_argument(
        '--experiment',
        type=str,
        default='xor',
        choices=list(EXPERIMENTS.keys()),
        help=f"The experiment to run. Choices: {list(EXPERIMENTS.keys())}"
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help="Disable Weights & Biases logging for this run."
    )
    parser.add_argument(
        '--load-model',
        type=str,
        default=None,
        help="Path to a pre-trained model file (.npz) to load for evaluation."
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        logging.warning(f"Unrecognized arguments will be ignored: {unknown}")
    train(args.experiment, no_wandb=args.no_wandb, load_model_path=args.load_model)

if __name__ == "__main__":
    main()