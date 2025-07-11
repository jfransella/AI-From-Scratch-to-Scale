"""
Training script for ADALINE (Adaptive Linear Neuron).

This module provides comprehensive training functionality for ADALINE with
proper logging, validation, experiment tracking, and educational visualizations.
"""

import logging
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path for imports
sys.path.append(str(Path(__file__).parent))

from src.config import config, ERROR_MESSAGES, SUCCESS_MESSAGES
from src.data_loader import create_data_loader
from src.model import create_adaline_model, ADALINEState
from src.evaluate import ADALINEEvaluator
from src.visualize import ADALINEVisualizer
from src.wandb_integration import initialize_wandb, finish_wandb, ADALINEWandbVisualizer


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_SAVE_PATH, 'training.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description='Train ADALINE (Adaptive Linear Neuron) model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument(
        '--data_source', 
        type=str, 
        default='synthetic',
        choices=['synthetic', 'iris'],
        help='Data source for training'
    )
    parser.add_argument(
        '--n_samples', 
        type=int, 
        default=1000,
        help='Number of samples for synthetic data'
    )
    parser.add_argument(
        '--n_features', 
        type=int, 
        default=2,
        help='Number of features for synthetic data'
    )
    parser.add_argument(
        '--noise', 
        type=float, 
        default=0.1,
        help='Noise level for synthetic data'
    )
    
    # Model parameters
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=0.01,
        help='Learning rate for gradient descent'
    )
    parser.add_argument(
        '--max_epochs', 
        type=int, 
        default=1000,
        help='Maximum number of training epochs'
    )
    parser.add_argument(
        '--convergence_threshold', 
        type=float, 
        default=1e-6,
        help='Loss threshold for convergence'
    )
    
    # Data preprocessing
    parser.add_argument(
        '--no_normalize', 
        action='store_true',
        help='Disable feature normalization'
    )
    parser.add_argument(
        '--no_bias', 
        action='store_true',
        help='Disable bias term addition'
    )
    
    # Experiment tracking
    parser.add_argument(
        '--use_wandb', 
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb_project', 
        type=str, 
        default='ai-from-scratch-adaline',
        help='Weights & Biases project name'
    )
    parser.add_argument(
        '--wandb_run_name', 
        type=str, 
        default='adaline-experiment',
        help='Weights & Biases run name'
    )
    
    # Output options
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='outputs',
        help='Output directory for results'
    )
    parser.add_argument(
        '--no_plots', 
        action='store_true',
        help='Disable plot generation'
    )
    parser.add_argument(
        '--plot_dpi', 
        type=int, 
        default=300,
        help='DPI for saved plots'
    )
    
    # Logging
    parser.add_argument(
        '--log_level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--log_every', 
        type=int, 
        default=10,
        help='Log training progress every N epochs'
    )
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> Any:
    """Create configuration object from command line arguments."""
    from src.config import ADALINEConfig
    
    # Create new config with command line overrides
    cfg = ADALINEConfig()
    
    # Update with command line arguments
    cfg.INPUT_SIZE = args.n_features
    cfg.LEARNING_RATE = args.learning_rate
    cfg.MAX_EPOCHS = args.max_epochs
    cfg.CONVERGENCE_THRESHOLD = args.convergence_threshold
    cfg.NORMALIZE_FEATURES = not args.no_normalize
    cfg.ADD_BIAS_TERM = not args.no_bias
    cfg.USE_WANDB = args.use_wandb
    cfg.WANDB_PROJECT_NAME = args.wandb_project
    cfg.WANDB_RUN_NAME = args.wandb_run_name
    cfg.OUTPUT_DIR = args.output_dir
    cfg.PLOTS_SAVE_PATH = os.path.join(args.output_dir, 'plots')
    cfg.LOGS_SAVE_PATH = os.path.join(args.output_dir, 'logs')
    cfg.SAVE_PLOTS = not args.no_plots
    cfg.PLOT_DPI = args.plot_dpi
    cfg.LOG_LEVEL = args.log_level
    cfg.LOG_EVERY_N_EPOCHS = args.log_every
    
    return cfg


class ADALINETrainer:
    """
    Comprehensive trainer for ADALINE model with experiment tracking and validation.
    
    This class orchestrates the complete training process including data loading,
    model training, evaluation, visualization, and experiment tracking.
    """
    
    def __init__(self, cfg: Any = None) -> None:
        """
        Initialize ADALINE trainer.
        
        Args:
            cfg: Configuration object
        """
        self.config = cfg if cfg is not None else config
        self.data_loader: Optional[Any] = None
        self.model: Optional[Any] = None
        self.evaluator: Optional[Any] = None
        self.visualizer: Optional[Any] = None
        self.training_state: Optional[Any] = None
        self.wandb_run: Optional[Any] = None
        self.wandb_visualizer: Optional[Any] = None
        
        # Create output directories
        self._create_output_directories()
        
        logger.info("ADALINE Trainer initialized")
    
    def _create_output_directories(self) -> None:
        """Create necessary output directories."""
        directories = [
            self.config.OUTPUT_DIR,
            self.config.PLOTS_SAVE_PATH,
            self.config.LOGS_SAVE_PATH
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def setup_components(self) -> None:
        """Initialize all training components."""
        logger.info("Setting up training components")
        
        # Initialize data loader
        self.data_loader = create_data_loader(self.config)
        logger.info("Data loader initialized")
        
        # Initialize model
        self.model = create_adaline_model(
            input_size=self.config.INPUT_SIZE + (1 if self.config.ADD_BIAS_TERM else 0),
            learning_rate=self.config.LEARNING_RATE,
            random_seed=self.config.RANDOM_SEED,
            cfg=self.config
        )
        logger.info("ADALINE model initialized")
        
        # Initialize evaluator
        self.evaluator = ADALINEEvaluator(self.config)
        logger.info("Evaluator initialized")
        
        # Initialize visualizer
        self.visualizer = ADALINEVisualizer(self.config)
        logger.info("Visualizer initialized")
        
        # Initialize W&B if enabled
        if self.config.USE_WANDB:
            self.wandb_run, self.wandb_visualizer = initialize_wandb(
                project_name=self.config.WANDB_PROJECT_NAME,
                config=self.config.to_dict(),
                enabled=True
            )
            logger.info("W&B integration initialized")
        else:
            self.wandb_visualizer = ADALINEWandbVisualizer(enabled=False)
            logger.info("W&B integration disabled")
    
    def load_data(self, data_source: str = 'synthetic', **kwargs) -> Dict[str, np.ndarray]:
        """
        Load and preprocess training data.
        
        Args:
            data_source: Source of data ('synthetic', 'iris')
            **kwargs: Additional arguments for data generation
            
        Returns:
            Dictionary containing data splits
        """
        logger.info(f"Loading data from source: {data_source}")
        
        try:
            data_splits = self.data_loader.load_and_preprocess(
                data_source=data_source, **kwargs)
            
            # Save data information
            data_info_path = os.path.join(self.config.OUTPUT_DIR, 'data_info.json')
            self.data_loader.save_data_info(data_splits, data_info_path)
            
            logger.info("Data loading completed successfully")
            return data_splits
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def train_model(self, 
                   data_splits: Dict[str, np.ndarray],
                   max_epochs: Optional[int] = None,
                   convergence_threshold: Optional[float] = None) -> ADALINEState:
        """
        Train the ADALINE model.
        
        Args:
            data_splits: Dictionary containing training and validation data
            max_epochs: Maximum number of training epochs
            convergence_threshold: Loss threshold for convergence
            
        Returns:
            Training state containing history and final parameters
        """
        logger.info("Starting ADALINE training")
        
        try:
            # Extract training and validation data
            X_train = data_splits['X_train']
            y_train = data_splits['y_train'].flatten()  # ADALINE expects 1D targets
            X_val = data_splits['X_val']
            y_val = data_splits['y_val'].flatten()
            
            # Custom training loop to log to W&B each epoch
            max_epochs_ = max_epochs or self.config.MAX_EPOCHS
            convergence_threshold_ = convergence_threshold or self.config.CONVERGENCE_THRESHOLD
            self.model._initialize_parameters()
            self.model.is_fitted = False
            self.model.training_loss = []
            self.model.validation_loss = []
            self.model.weight_history = []
            self.model.bias_history = []
            best_loss = float('inf')
            patience_counter = 0
            max_patience = self.config.MAX_ITERATIONS_WITHOUT_IMPROVEMENT
            for epoch in range(max_epochs_):
                y_pred = self.model.forward(X_train)
                train_loss = self.model._compute_loss(y_train, y_pred)
                self.model.training_loss.append(train_loss)
                val_loss = None
                if X_val is not None and y_val is not None:
                    y_val_pred = self.model.forward(X_val)
                    val_loss = self.model._compute_loss(y_val, y_val_pred)
                    self.model.validation_loss.append(val_loss)
                self.model.weight_history.append(self.model.weights.copy())
                self.model.bias_history.append(self.model.bias)
                # Log to W&B if enabled
                if self.wandb_visualizer and self.wandb_visualizer.enabled:
                    metrics = {"train_loss": train_loss}
                    if val_loss is not None:
                        metrics["val_loss"] = val_loss
                    self.wandb_visualizer.log_training_progress(metrics, step=epoch)
                # Log progress
                if epoch % self.config.LOG_EVERY_N_EPOCHS == 0:
                    log_msg = f"Epoch {epoch+1}/{max_epochs_}: train_loss={train_loss:.6f}"
                    if val_loss is not None:
                        log_msg += f", val_loss={val_loss:.6f}"
                    logger.info(log_msg)
                current_loss = val_loss if val_loss is not None else train_loss
                if current_loss < convergence_threshold_:
                    logger.info(SUCCESS_MESSAGES['training_converged'].format(epoch + 1))
                    break
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                weight_gradients, bias_gradient = self.model._compute_gradients(X_train, y_train, y_pred)
                self.model._update_parameters(weight_gradients, bias_gradient)
            self.model.is_fitted = True
            self.training_state = ADALINEState(
                weights=self.model.weights.copy(),
                bias=self.model.bias,
                training_loss=np.array(self.model.training_loss),
                validation_loss=np.array(self.model.validation_loss) if self.model.validation_loss else np.array([]),
                training_accuracy=np.array([]),
                validation_accuracy=np.array([]),
                weight_history=np.array(self.model.weight_history),
                bias_history=np.array(self.model.bias_history),
                convergence_epoch=len(self.model.training_loss),
                final_loss=self.model.training_loss[-1] if self.model.training_loss else None
            )
            logger.info("Training completed successfully")
            return self.training_state
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def evaluate_model(self, data_splits: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Args:
            data_splits: Dictionary containing test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating trained model")
        
        try:
            X_test = data_splits['X_test']
            y_test = data_splits['y_test'].flatten()
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Evaluate predictions
            metrics = self.evaluator.evaluate_predictions(y_test, y_pred)
            
            # Compute R² score
            r2_score = self.model.score(X_test, y_test)
            metrics['r2_score'] = r2_score
            
            logger.info(f"Evaluation results: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
    
    def create_visualizations(self, 
                            data_splits: Dict[str, np.ndarray],
                            save_plots: bool = True) -> Dict[str, str]:
        """
        Create comprehensive visualizations for the training process.
        
        Args:
            data_splits: Dictionary containing all data splits
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        logger.info("Creating training visualizations")
        
        try:
            plot_paths = {}
            
            # Training progress plots
            if self.training_state is not None:
                progress_plot_path = self.visualizer.plot_training_progress(
                    self.training_state, save=save_plots)
                plot_paths['training_progress'] = progress_plot_path
                
                # Weight evolution plots
                weight_plot_path = self.visualizer.plot_weight_evolution(
                    self.training_state, save=save_plots)
                plot_paths['weight_evolution'] = weight_plot_path
            
            # Decision boundary plots
            boundary_plot_path = self.visualizer.plot_decision_boundary(
                self.model, data_splits, save=save_plots)
            plot_paths['decision_boundary'] = boundary_plot_path
            
            # Model comparison plots
            comparison_plot_path = self.visualizer.plot_model_comparison(
                self.model, data_splits, save=save_plots)
            plot_paths['model_comparison'] = comparison_plot_path
            
            logger.info("Visualizations created successfully")
            return plot_paths
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model (optional)
            
        Returns:
            Path where model was saved
        """
        if not self.model.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        filepath = filepath or self.config.MODEL_SAVE_PATH
        self.model.save_model(filepath)
        
        logger.info(f"Model saved to: {filepath}")
        return filepath
    
    def log_experiment(self, 
                      data_splits: Dict[str, np.ndarray],
                      evaluation_metrics: Dict[str, float],
                      plot_paths: Dict[str, str]) -> None:
        """
        Log experiment results and metadata.
        
        Args:
            data_splits: Dictionary containing data splits
            evaluation_metrics: Dictionary of evaluation metrics
            plot_paths: Dictionary of plot file paths
        """
        logger.info("Logging experiment results")
        
        try:
            # Create experiment summary
            experiment_summary = {
                'model_type': 'ADALINE',
                'config': self.config.to_dict(),
                'data_info': self.data_loader.get_data_info(data_splits),
                'training_info': {
                    'final_loss': self.training_state.final_loss if self.training_state else None,
                    'convergence_epoch': self.training_state.convergence_epoch if self.training_state else None,
                    'total_epochs': len(self.training_state.training_loss) if self.training_state else 0
                },
                'evaluation_metrics': evaluation_metrics,
                'plot_paths': plot_paths,
                'model_parameters': self.model.get_parameters()
            }
            
            # Save experiment summary locally
            import json
            summary_path = os.path.join(self.config.OUTPUT_DIR, 'experiment_summary.json')
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            experiment_summary = convert_numpy(experiment_summary)
            
            with open(summary_path, 'w') as f:
                json.dump(experiment_summary, f, indent=2)
            
            logger.info(f"Experiment summary saved to: {summary_path}")
            
            # Log to W&B if enabled
            if self.wandb_visualizer and self.wandb_visualizer.enabled:
                # Log training results with visualizations
                self.wandb_visualizer.log_training_results(
                    model=self.model,
                    training_state=self.training_state,
                    X=data_splits['X_train'],
                    y=data_splits['y_train'],
                    predictions=self.model.predict(data_splits['X_test']),
                    metrics=evaluation_metrics
                )
                
                # Log experiment summary
                self.wandb_visualizer.log_experiment_results(
                    "experiment_summary", experiment_summary
                )
                
                logger.info("Experiment results logged to W&B")
            
        except Exception as e:
            logger.error(f"Error logging experiment: {e}")
            raise
    
    def run_complete_training(self, 
                            data_source: str = 'synthetic',
                            **kwargs) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Args:
            data_source: Source of data ('synthetic', 'iris')
            **kwargs: Additional arguments for data generation
            
        Returns:
            Dictionary containing all training results
        """
        logger.info("Starting complete ADALINE training pipeline")
        
        try:
            # Setup components
            self.setup_components()
            
            # Ensure components are initialized
            if self.data_loader is None or self.model is None or self.evaluator is None or self.visualizer is None:
                raise RuntimeError("Training components not properly initialized")
            
            # Load data
            data_splits = self.load_data(data_source, **kwargs)
            
            # Train model
            training_state = self.train_model(data_splits)
            
            # Evaluate model
            evaluation_metrics = self.evaluate_model(data_splits)
            
            # Create visualizations
            plot_paths = self.create_visualizations(data_splits)
            
            # Save model
            model_path = self.save_model()
            
            # Log experiment
            self.log_experiment(data_splits, evaluation_metrics, plot_paths)
            
            results = {
                'training_state': training_state,
                'evaluation_metrics': evaluation_metrics,
                'plot_paths': plot_paths,
                'model_path': model_path,
                'data_splits': data_splits
            }
            
            logger.info("Complete training pipeline finished successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise


def main():
    """Main training function."""
    logger.info("Starting ADALINE training script")
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Create config from arguments
        cfg = create_config_from_args(args)
        
        # Create trainer
        trainer = ADALINETrainer(cfg)
        
        # Run complete training pipeline
        results = trainer.run_complete_training(
            data_source=args.data_source,
            n_samples=args.n_samples,
            n_features=args.n_features,
            noise=args.noise,
            problem_type='regression'
        )
        
        # Print summary
        print("\n" + "="*50)
        print("ADALINE TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Final Loss: {results['training_state'].final_loss:.6f}")
        print(f"Convergence Epoch: {results['training_state'].convergence_epoch}")
        print(f"R² Score: {results['evaluation_metrics']['r2_score']:.4f}")
        print(f"Model saved to: {results['model_path']}")
        print(f"Plots saved to: {results['plot_paths']}")
        if trainer.config.USE_WANDB:
            print("Experiment logged to Weights & Biases")
        print("="*50)
        
        logger.info("Training script completed successfully")
        
    except Exception as e:
        logger.error(f"Training script failed: {e}")
        raise
    finally:
        # Cleanup W&B
        if 'trainer' in locals() and hasattr(trainer, 'wandb_run'):
            finish_wandb(trainer.wandb_run)


if __name__ == "__main__":
    main() 