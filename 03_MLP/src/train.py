# -*- coding: utf-8 -*-
"""Main training and evaluation script for the Multi-Layer Perceptron (MLP) model.

This script serves as the main orchestrator for ML experiments, demonstrating professional
ML development practices including:
- Configuration-driven experiments 
- Comprehensive logging and monitoring
- Reproducible experimental setup
- Integration with experiment tracking (Weights & Biases)
- Modular evaluation and visualization

Educational Context:
    This training script exemplifies the "separation of concerns" principle in ML:
    - Data loading is handled by data_loader.py
    - Model architecture is defined in model.py  
    - Evaluation metrics are computed by evaluate.py
    - Visualizations are created by visualize.py
    - All configuration is centralized in config.py
    
    This modular approach makes the codebase maintainable and experiments reproducible.

Example usage:
    # Run different experiments
    python -m src.train --experiment xor
    python -m src.train --experiment mnist-multiclass
    python -m src.train --experiment mnist-failure-test
    
    # Disable W&B logging
    python -m src.train --experiment xor --no-wandb
    
    # Evaluate a saved model
    python -m src.train --experiment mnist-multiclass --load-model outputs/models/model.npz
"""

import argparse
import logging
import os
import sys
from datetime import datetime
import sys
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import wandb

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from config import EXPERIMENTS, WANDB_PROJECT_NAME, DEFAULT_RANDOM_SEED
from model import MLP
from evaluate import evaluate_model, print_evaluation_report, calculate_model_robustness
from visualize import Visualizer

logger = logging.getLogger(__name__)


def train(experiment: str, no_wandb: bool = False, load_model_path: Optional[str] = None) -> None:
    """Orchestrates a complete ML experiment: data loading, training, evaluation, and visualization.
    
    Educational Context:
        This function demonstrates the standard ML workflow:
        1. **Data Preparation**: Load and validate data for the experiment
        2. **Model Setup**: Initialize model architecture and hyperparameters
        3. **Training**: Fit model parameters using gradient descent
        4. **Evaluation**: Assess performance using comprehensive metrics
        5. **Visualization**: Create plots for understanding model behavior
        6. **Artifact Management**: Save models and results for reproducibility
    
    Args:
        experiment: Name of the experiment to run (must be defined in EXPERIMENTS config)
        no_wandb: Whether to disable Weights & Biases logging for this run
        load_model_path: Path to pre-trained model file for evaluation-only mode
        
    Raises:
        ValueError: If experiment name is not found in configuration
        FileNotFoundError: If load_model_path is provided but file doesn't exist
        RuntimeError: If data loading or model training fails
    """
    logger.info(f"🚀 Starting MLP Experiment: '{experiment}'")

    # === STEP 1: VALIDATE EXPERIMENT CONFIGURATION ===
    if experiment not in EXPERIMENTS:
        available_experiments = list(EXPERIMENTS.keys())
        logger.error(f"❌ Unknown experiment: '{experiment}'. Available: {available_experiments}")
        raise ValueError(f"Unknown experiment: '{experiment}'. Check config.py for available experiments.")

    exp_config = EXPERIMENTS[experiment]
    logger.info(f"📋 Experiment: {exp_config.get('description', experiment)}")
    
    # === STEP 2: DATA LOADING ===
    logger.info(f"📁 Loading data for '{experiment}' experiment...")
    
    try:
        # Load data using the experiment's configured data loader
        data_tuple = exp_config["data_loader"]()
        
        # Handle different data loader return formats
        if len(data_tuple) == 4:
            X_train, y_train, X_test, y_test = data_tuple
            logger.info(f"✅ Data loaded: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
        else:
            X_train, y_train = data_tuple
            X_test, y_test = X_train, y_train  # Use training data for evaluation (simple problems)
            logger.info(f"✅ Data loaded: {X_train.shape[0]} samples (train=test)")
            
        # Validate data integrity
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError("Loaded training data is empty")
        if X_test.size == 0 or y_test.size == 0:
            raise ValueError("Loaded test data is empty")
            
    except Exception as e:
        logger.error(f"❌ Failed to load data for experiment '{experiment}': {e}")
        raise RuntimeError(f"Data loading failed: {e}")

    # === STEP 3: REPRODUCIBILITY SETUP ===
    # Set random seed for reproducible experiments
    np.random.seed(DEFAULT_RANDOM_SEED)
    logger.info(f"🎲 Set random seed to {DEFAULT_RANDOM_SEED} for reproducible results")

    # === STEP 4: EXPERIMENT TRACKING SETUP ===
    wandb_mode = "disabled" if no_wandb else "online"
    wandb.init(
        mode=wandb_mode,
        project=WANDB_PROJECT_NAME,
        config={
            "learning_rate": exp_config["learning_rate"],
            "epochs": exp_config["epochs"],
            "hidden_size": exp_config["hidden_size"],
            "experiment_type": experiment,
            "description": exp_config.get("description", ""),
            "random_seed": DEFAULT_RANDOM_SEED,
        }
    )
    
    if no_wandb:
        logger.info("📊 Weights & Biases logging is disabled for this run")
    else:
        logger.info(f"📊 W&B run '{wandb.run.name}' initialized for experiment tracking")

    # Initialize visualization manager
    visualizer = Visualizer(wandb, enabled=(not no_wandb))

    # === STEP 5: MODEL INITIALIZATION ===
    # Get hyperparameters from W&B config (supports hyperparameter sweeps)
    learning_rate = wandb.config.learning_rate
    epochs = wandb.config.epochs
    hidden_size = wandb.config.hidden_size

    model = MLP(
        input_size=exp_config["input_size"],
        hidden_size=hidden_size,
        output_size=exp_config["output_size"],
        learning_rate=learning_rate,
        epochs=epochs,
        random_seed=DEFAULT_RANDOM_SEED,
    )
    
    logger.info(f"🧠 Model initialized: {model}")

    # === STEP 6: TRAINING OR MODEL LOADING ===
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    if load_model_path:
        # Evaluation-only mode: load pre-trained model
        if not os.path.exists(load_model_path):
            raise FileNotFoundError(f"Model file not found: {load_model_path}")
        
        logger.info(f"📥 Loading pre-trained model from {load_model_path}")
        model.load_model(load_model_path)
        
        # Create descriptive run name for evaluation
        base_name = os.path.basename(load_model_path).replace('.npz', '')
        run_name = f"eval-{base_name}-{timestamp}"
        logger.info(f"🏷️  Evaluation mode: {run_name}")
        
    else:
        # Training mode: train new model from scratch
        run_name = f"{experiment}-{timestamp}-lr{learning_rate:.4f}-hs{hidden_size}-ep{epochs}"
        logger.info(f"🎯 Training new model: {run_name}")
        
        # Start training process
        logger.info(f"🔥 Starting training with backpropagation...")
        model.fit(X_train, y_train)
        logger.info(f"✅ Training completed!")

        # Save the newly trained model for future use
        model_dir = "outputs/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{run_name}.npz")
        model.save_model(model_path)
        logger.info(f"💾 Model saved to: {model_path}")

    # Update W&B run name for better organization
    if not no_wandb:
        wandb.run.name = run_name
        logger.info(f"📊 W&B run name updated to: {run_name}")

    # === STEP 7: COMPREHENSIVE EVALUATION ===
    logger.info(f"📈 Starting comprehensive model evaluation...")
    
    # Perform detailed evaluation using the evaluate module
    evaluation_results = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_names=exp_config.get("class_names")
    )
    
    # Display detailed evaluation report
    print_evaluation_report(evaluation_results)
    
    # Log key metrics to experiment tracking
    if not no_wandb:
        wandb.summary.update({
            "final_accuracy": evaluation_results['accuracy'],
            "final_precision": evaluation_results['precision'],
            "final_recall": evaluation_results['recall'],
            "final_f1_score": evaluation_results['f1_score'],
            "test_samples": evaluation_results['n_samples'],
            "model_parameters": (
                model.W1.size + model.b1.size + model.W2.size + model.b2.size
            )
        })
    
    # === STEP 8: ROBUSTNESS ANALYSIS (for failure experiments) ===
    if "failure" in experiment and len(data_tuple) == 4:
        logger.info(f"🔬 Performing robustness analysis...")
        
        # Load original (unmodified) test data for comparison
        from data_loader import load_mnist_multiclass_data
        _, _, X_test_original, _ = load_mnist_multiclass_data(return_test_set=True)
        
        # Calculate robustness metrics
        robustness_metrics = calculate_model_robustness(
            model=model,
            X_original=X_test_original,
            X_modified=X_test,  # X_test contains the shifted/modified images
            y_test=y_test
        )
        
        logger.info(f"🛡️  Robustness Score: {robustness_metrics['robustness_score']:.4f}")
        logger.info(f"📉 Accuracy Drop: {robustness_metrics['accuracy_drop']:.4f}")
        
        # Log robustness metrics
        if not no_wandb:
            wandb.summary.update({
                "robustness_score": robustness_metrics['robustness_score'],
                "accuracy_drop": robustness_metrics['accuracy_drop'],
                "relative_accuracy_drop": robustness_metrics['relative_accuracy_drop']
            })

    # === STEP 9: VISUALIZATION ===
    logger.info(f"📊 Generating visualizations...")
    
    class_names = exp_config.get("class_names")
    predictions = model.predict(X_test)
    
    # Create and log all relevant visualizations
    visualizer.log_all(
        model=model,
        X=X_test,
        y=y_test,
        predictions=predictions,
        class_names=class_names
    )

    # === STEP 10: EXPERIMENT COMPLETION ===
    wandb.finish()
    logger.info(f"🎉 Experiment '{experiment}' completed successfully!")
    logger.info(f"📊 Final accuracy: {evaluation_results['accuracy']:.4f}")
    if model.losses:
        logger.info(f"📉 Final training loss: {model.losses[-1]:.6f}")


def main() -> None:
    """Sets up logging, parses command-line arguments, and orchestrates the training process.
    
    Educational Context:
        This main function demonstrates best practices for ML script organization:
        1. **Output Directory Management**: Ensure required directories exist
        2. **Logging Configuration**: Set up both file and console logging
        3. **Argument Parsing**: Use argparse for clean command-line interface
        4. **Error Handling**: Graceful failure with informative error messages
        5. **Exit Codes**: Proper exit codes for automation and CI/CD
    """
    # === STEP 1: SETUP OUTPUT DIRECTORIES ===
    # Create necessary output directories following project structure
    output_dirs = ["outputs/logs", "outputs/models", "outputs/plots"]
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # === STEP 2: CONFIGURE LOGGING ===
    # Setup dual logging: console output for real-time monitoring, file for persistence
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outputs/logs/mlp_training.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("🚀 MLP Training Script Started")
    logger.info(f"📁 Output directories created: {', '.join(output_dirs)}")

    # === STEP 3: COMMAND-LINE ARGUMENT PARSING ===
    parser = argparse.ArgumentParser(
        description="Train and evaluate Multi-Layer Perceptron models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --experiment xor                    # Train XOR classifier
  %(prog)s --experiment mnist-multiclass       # Train MNIST classifier  
  %(prog)s --experiment xor --no-wandb         # Train without W&B logging
  %(prog)s --load-model outputs/models/model.npz  # Evaluate saved model
        """
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default='xor',
        choices=list(EXPERIMENTS.keys()),
        help=f"Experiment to run. Available: {list(EXPERIMENTS.keys())} (default: xor)"
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help="Disable Weights & Biases logging for this run"
    )
    parser.add_argument(
        '--load-model',
        type=str,
        default=None,
        metavar='PATH',
        help="Path to pre-trained model file (.npz) for evaluation only"
    )
    
    # Parse arguments and handle unknown arguments gracefully
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f"⚠️  Unrecognized arguments ignored: {unknown}")
    
    # Log parsed arguments for debugging
    logger.info(f"🔧 Experiment: {args.experiment}")
    logger.info(f"🔧 W&B Logging: {'disabled' if args.no_wandb else 'enabled'}")
    if args.load_model:
        logger.info(f"🔧 Model Loading: {args.load_model}")
    
    # === STEP 4: EXECUTE TRAINING PIPELINE ===
    try:
        train(
            experiment=args.experiment, 
            no_wandb=args.no_wandb, 
            load_model_path=args.load_model
        )
        logger.info("✅ Training pipeline completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Training interrupted by user (Ctrl+C)")
        sys.exit(130)  # Standard exit code for script terminated by Control-C
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        logger.error("💡 Check logs for detailed error information")
        sys.exit(1)  # Exit with error code

if __name__ == "__main__":
    main()