#!/usr/bin/env python3
"""
Training script with PerceptronVisualizer integration
===================================================

This script runs Perceptron training experiments using our new
PerceptronVisualizer for generating local visualizations.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import numpy as np
from config import EXPERIMENTS
from model import Perceptron
from visualize import PerceptronVisualizer

def train_with_visualizer(experiment: str) -> None:
    """Run training with our new PerceptronVisualizer.
    
    Args:
        experiment: The name of the experiment to run
    """
    logging.info("--- Starting Perceptron Training with Visualizer: '%s' experiment ---", experiment)

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

    # --- 2. Initialize model and visualizer ---
    lr = exp_config["learning_rate"]
    epochs = exp_config["epochs"]
    
    logging.info("Initializing Perceptron with LR=%f and Epochs=%d.", lr, epochs)
    perceptron = Perceptron(learning_rate=lr, n_iters=epochs, logger=logging.getLogger())
    
    # Initialize our visualizer
    visualizer = PerceptronVisualizer(save_dir="outputs/plots")
    
    # --- 3. Train the model ---
    logging.info("Starting training...")
    perceptron.fit(features, y)
    logging.info("Training complete.")

    # --- 4. Evaluate ---
    predictions = perceptron.predict(features)
    accuracy = (predictions == y).mean()
    logging.info("Final training accuracy: %.4f", accuracy)

    # --- 5. Generate visualizations ---
    logging.info("Generating visualizations...")
    
    # Get errors per epoch from the model
    errors_per_epoch = perceptron.errors_per_epoch if hasattr(perceptron, 'errors_per_epoch') else []
    
    # Generate all visualizations
    class_names = exp_config.get("class_names", ['Class 0', 'Class 1'])
    
    visualizations = visualizer.generate_all_visualizations(
        model=perceptron,
        features=features,
        y=y,
        y_pred=predictions,
        errors_per_epoch=errors_per_epoch,
        class_names=class_names
    )
    
    logging.info(f"Generated {len(visualizations)} visualizations:")
    for viz_name, fig in visualizations.items():
        if fig is not None:
            logging.info(f"  ✓ {viz_name}")
        else:
            logging.info(f"  ✗ {viz_name} (failed)")
    
    # --- 6. Generate performance insights ---
    if len(predictions) > 0:
        tn, fp, fn, tp = 0, 0, 0, 0
        for true, pred in zip(y, predictions):
            if true == 1 and pred == 1:
                tp += 1
            elif true == 1 and pred == 0:
                fn += 1
            elif true == 0 and pred == 1:
                fp += 1
            else:
                tn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        insights = visualizer._generate_performance_insights(accuracy, precision, recall)
        logging.info("\n" + insights)
    
    logging.info("--- Experiment '%s' Finished ---", experiment)

def main():
    """Main entry point."""
    # --- Logging Setup ---
    os.makedirs("outputs/logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outputs/logs/training_with_visualizer.log'),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description="Run Perceptron training with visualizations.")
    parser.add_argument(
        '--experiment',
        type=str,
        default='and',
        choices=list(EXPERIMENTS.keys()),
        help="The experiment to run. Choices: %s" % list(EXPERIMENTS.keys())
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        logging.warning("Unrecognized arguments will be ignored: %s", unknown)
    
    train_with_visualizer(args.experiment)

if __name__ == "__main__":
    main() 