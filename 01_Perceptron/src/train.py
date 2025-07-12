# -*- coding: utf-8 -*-
"""Main training and evaluation script for the Perceptron model.

This script serves as the main entry point for running experiments. It handles:
- Data loading based on the selected experiment from `config.py`.
- Model initialization and training.
- Integration with Weights & Biases (wandb) for logging metrics, parameters,
  and visualizations.
- Support for hyperparameter sweeps using wandb agents.

Example usage:
    # Run a single experiment
    python -m src.train --experiment and

    # Run a sweep (requires a sweep.yaml file and pre-initialization)
    wandb agent <sweep_id>
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import wandb
from .config import WANDB_PROJECT_NAME, EXPERIMENTS
from .model import Perceptron
from .wandb_integration import PerceptronWandbVisualizer

# --- Environment Verification ---
def _verify_virtual_environment() -> None:
    """
    Verifies that the script is running in the project's virtual environment.

    If the interpreter path does not match the expected path within the `.venv`
    directory, it prints an error and exits the script with a non-zero status code.
    """
    # Determine the expected path of the Python executable in the virtual env
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if sys.platform == "win32":
        expected_path = os.path.join(project_root, '.venv', 'Scripts', 'python.exe')
    else:
        expected_path = os.path.join(project_root, '.venv', 'bin', 'python')

    # Normalize for case-insensitive comparison
    expected_executable = os.path.normcase(expected_path)
    current_executable = os.path.normcase(sys.executable)

    if current_executable != expected_executable:
        print(
            f"Error: Script is not running in the correct virtual environment.\n"
            f"  Current Interpreter: {sys.executable}\n"
            f"  Expected Interpreter: {expected_path}",
            file=sys.stderr,
        )
        activation_cmd = (
            ".\\.venv\\Scripts\\activate" if sys.platform == "win32" 
            else "source .venv/bin/activate"
        )
        print(f"\nPlease activate the virtual environment: `{activation_cmd}`", file=sys.stderr)
        sys.exit(1)

_verify_virtual_environment()

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
# --- Logging Setup ---
def train(experiment: str, no_wandb: bool = False) -> None:
    """Orchestrates a single training and evaluation run for a given experiment.

    This function performs the following steps:
    1.  Loads the dataset specified by the `experiment` name from the configuration.
    2.  Initializes a Weights & Biases run (unless disabled).
    3.  Retrieves hyperparameters from the W&B config, which allows for sweeps.
    4.  Initializes and trains the Perceptron model.
    5.  Evaluates the model's final accuracy on the training data.
    6.  Uses the `PerceptronWandbVisualizer` to generate and log all relevant plots.
    7.  Finishes the W&B run.

    Args:
        experiment: The name of the experiment to run (e.g., 'and', 'mnist').
                   Must be a key in the `EXPERIMENTS` dictionary in `config.py`.
        no_wandb: If True, disables all Weights & Biases logging.
    """
    logging.info("--- Starting Perceptron Training: '%s' experiment ---", experiment)

    # --- 1. Prepare Experiment ---
    if experiment not in EXPERIMENTS:
        logging.error(
            "Unknown experiment: %s. Check config.py for available experiments.", 
            experiment
        )
        return

    exp_config = EXPERIMENTS[experiment]
    logging.info("Loading data for '%s' experiment...", experiment)
    features, y = exp_config["data_loader"]()
    logging.info(
        "Data loaded successfully. Found %d samples with %d features.",
        features.shape[0], features.shape[1]
    )

    # --- Setup W&B ---
    wandb_mode = "disabled" if no_wandb else "online"

    wandb.init(
        mode=wandb_mode,
        project=WANDB_PROJECT_NAME,
        config={
            "learning_rate": exp_config["learning_rate"],
            "epochs": exp_config["epochs"],
            "experiment_type": experiment,
        }
    )
    if no_wandb:
        logging.info("Weights & Biases logging is disabled for this run.")
    else:
        assert wandb.run is not None  # type: ignore
        logging.info("Weights & Biases run '%s' initialized.", wandb.run.name)

    # Initialize the wandb visualizer
    visualizer = PerceptronWandbVisualizer(wandb.run) if not no_wandb else None

    # Get hyperparameters from the W&B config.
    # These will be the defaults for a single run, or provided by the sweep agent.
    lr = wandb.config.learning_rate
    epochs = wandb.config.epochs

    # Set a more descriptive run name, especially for sweeps
    if not no_wandb:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{experiment}-{timestamp}-lr_{lr:.5f}-ep_{epochs}"
        assert wandb.run is not None  # type: ignore
        wandb.run.name = run_name
        logging.info("W&B run name updated to: %s", run_name)

    # --- 2. Initialize and train the model ---
    logging.info("Initializing Perceptron with LR=%f and Epochs=%d.", lr, epochs)
    perceptron = Perceptron(learning_rate=lr, n_iters=epochs, logger=logging.getLogger())
    perceptron.fit(features, y)
    logging.info("Training complete.")

    # --- 3. Evaluate ---
    predictions = perceptron.predict(features)
    # Assuming y from data loader is already 0s and 1s
    accuracy = (predictions == y).mean()
    logging.info("Final training accuracy: %.4f", accuracy)

    # Log final accuracy to W&B summary for easy comparison
    wandb.summary["final_accuracy"] = accuracy

    # --- 4. Visualize ---
    if visualizer:
        class_names = exp_config.get("class_names")
        visualizer.log_training_results(
            model=perceptron,
            X=features,
            y=y,
            predictions=predictions,
            class_names=class_names
        )

    wandb.finish()
    logging.info("--- Experiment '%s' Finished ---", experiment)


def main():
    """Sets up logging, parses command-line arguments, and starts the training process.

    This function acts as the main entry point when the script is executed. It
    configures the root logger, defines the command-line interface for selecting
    an experiment, and then calls the `train` function with the parsed arguments.
    """
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

    parser = argparse.ArgumentParser(description="Run a Perceptron training experiment.")
    parser.add_argument(
        '--experiment',
        type=str,
        default='and',
        choices=list(EXPERIMENTS.keys()),
        help="The experiment to run. Choices: %s" % list(EXPERIMENTS.keys())
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help="Disable Weights & Biases logging for this run."
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        logging.warning("Unrecognized arguments will be ignored: %s", unknown)
    train(args.experiment, no_wandb=args.no_wandb)

if __name__ == "__main__":
    main()
