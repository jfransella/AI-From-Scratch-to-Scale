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
from typing import Optional, Tuple

import numpy as np
import wandb

from src.config import EXPERIMENTS, WANDB_PROJECT_NAME
from src.model import MLP
from src.evaluate import evaluate_model, print_evaluation_report, calculate_model_robustness
from src.visualize import Visualizer

logger = logging.getLogger(__name__)


def train(experiment: str, no_wandb: bool = False, load_model_path: Optional[str] = None) -> None:
    """Orchestrates a single training and evaluation run for a given experiment.
    
    Args:
        experiment: Name of the experiment to run (must be in EXPERIMENTS config)
        no_wandb: Whether to disable Weights & Biases logging
        load_model_path: Path to pre-trained model file for evaluation only
        
    Raises:
        ValueError: If experiment name is not found in config
        FileNotFoundError: If load_model_path is provided but file doesn't exist
    """
    logger.info(f"--- Starting MLP Training: '{experiment}' experiment ---")

    # --- 1. Prepare Experiment ---
    if experiment not in EXPERIMENTS:
        available_experiments = list(EXPERIMENTS.keys())
        logger.error(f"Unknown experiment: {experiment}. Available: {available_experiments}")
        raise ValueError(f"Unknown experiment: {experiment}. Check config.py.")

    exp_config = EXPERIMENTS[experiment]
    logger.info(f"Loading data for '{experiment}' experiment...")
    
    try:
        data_tuple = exp_config["data_loader"]()
        if len(data_tuple) == 4:
            X_train, y_train, X_test, y_test = data_tuple
            logger.info(f"Data loaded successfully. Found {X_train.shape[0]} training samples and {X_test.shape[0]} test samples.")
        else:
            X_train, y_train = data_tuple
            X_test, y_test = X_train, y_train  # Fallback for old data loaders
            logger.info(f"Data loaded successfully. Found {X_train.shape[0]} samples.")
    except Exception as e:
        logger.error(f"Failed to load data for experiment '{experiment}': {e}")
        raise

    # Validate loaded data
    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Loaded data is empty")

    # Set random seed for reproducibility
    np.random.seed(42)
    logger.info("Set random seed to 42 for reproducibility")

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
        logger.info("Weights & Biases logging is disabled for this run.")
    else:
        logger.info(f"Weights & Biases run '{wandb.run.name}' initialized.")

    visualizer = Visualizer(wandb, enabled=(not no_wandb))

    # Get hyperparameters from the W&B config (allows for sweeps)
    learning_rate = wandb.config.learning_rate
    epochs = wandb.config.epochs
    hidden_size = wandb.config.hidden_size

    model = MLP(
        input_size=exp_config["input_size"],
        hidden_size=hidden_size,
        output_size=exp_config["output_size"],
        learning_rate=learning_rate,
        epochs=epochs,
        random_seed=42,  # For reproducibility
        wandb_run=wandb,
    )

    # --- 2. Load or Train Model ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if load_model_path:
        if not os.path.exists(load_model_path):
            raise FileNotFoundError(f"Model file not found: {load_model_path}")
        
        logger.info(f"Loading pre-trained model from {load_model_path}")
        model.load_model(load_model_path)
        # Use the loaded model's name for the run, with an 'eval-' prefix
        base_name = os.path.basename(load_model_path).replace('.npz', '')
        run_name = f"eval-{base_name}-{timestamp}"
    else:
        # Set a descriptive run name for a new training run
        run_name = f"{experiment}-{timestamp}-lr_{learning_rate:.4f}-hs_{hidden_size}-ep_{epochs}"
        logger.info(f"Initializing MLP for new training run: {run_name}")
        model.fit(X_train, y_train)
        logger.info("Training complete.")

        # Save the newly trained model
        model_dir = "outputs/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{run_name}.npz")
        model.save_model(model_path)

    if not no_wandb:
        wandb.run.name = run_name
        logger.info(f"W&B run name updated to: {run_name}")

    # --- 3. Evaluate ---
    logger.info("Starting comprehensive evaluation...")
    
    # Comprehensive evaluation using the new evaluate module
    evaluation_results = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_names=exp_config.get("class_names")
    )
    
    # Print detailed evaluation report
    print_evaluation_report(evaluation_results)
    
    # Log main metrics to wandb
    if not no_wandb:
        wandb.summary.update({
            "final_eval_accuracy": evaluation_results['accuracy'],
            "final_eval_precision": evaluation_results['precision'],
            "final_eval_recall": evaluation_results['recall'],
            "final_eval_f1": evaluation_results['f1_score']
        })
    
    # Special robustness test for failure experiments
    if "failure" in experiment and len(data_tuple) == 4:
        logger.info("Performing robustness analysis...")
        # Load original test data for comparison
        from src.data_loader import load_mnist_multiclass_data
        _, _, X_test_original, _ = load_mnist_multiclass_data(return_test_set=True)
        
        robustness_metrics = calculate_model_robustness(
            model=model,
            X_original=X_test_original,
            X_modified=X_test,  # X_test is the shifted version
            y_test=y_test
        )
        
        logger.info(f"Robustness Score: {robustness_metrics['robustness_score']:.4f}")
        
        if not no_wandb:
            wandb.summary.update({
                "robustness_score": robustness_metrics['robustness_score'],
                "accuracy_drop": robustness_metrics['accuracy_drop']
            })

    # --- 4. Visualize ---
    class_names = exp_config.get("class_names")
    predictions = model.predict(X_test)
    visualizer.log_all(
        model=model,
        X=X_test,
        y=y_test,
        predictions=predictions,
        class_names=class_names
    )

    wandb.finish()
    logger.info(f"--- Experiment '{experiment}' Finished ---")


def main() -> None:
    """Sets up logging, parses arguments, and starts the training process."""
    # Create output directories
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
        logger.warning(f"Unrecognized arguments will be ignored: {unknown}")
    
    try:
        train(args.experiment, no_wandb=args.no_wandb, load_model_path=args.load_model)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()