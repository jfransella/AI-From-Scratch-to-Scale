"""
Data loading and preprocessing utilities for ADALINE (Adaptive Linear Neuron).

This module provides comprehensive data loading, preprocessing, and validation
functions for ADALINE training and evaluation, following the project's standards.
"""

import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import config, ERROR_MESSAGES, SUCCESS_MESSAGES


logger = logging.getLogger(__name__)


class ADALINEDataLoader:
    """
    Data loader for ADALINE model with comprehensive preprocessing and validation.
    
    This class handles data generation, preprocessing, validation, and splitting
    for ADALINE training and evaluation. It supports both synthetic and real datasets.
    """
    
    def __init__(self, cfg: Any = None) -> None:
        """
        Initialize the ADALINE data loader.
        
        Args:
            cfg: Configuration object containing data parameters.
                 If None, uses default config.
        """
        self.config = cfg if cfg is not None else config
        self.scaler = None
        self.is_fitted = False
        self._validate_config()
        
        # Set random seed for reproducibility
        np.random.seed(self.config.RANDOM_SEED)
        logger.info("ADALINE DataLoader initialized with config: %s", 
                   {k: v for k, v in self.config.to_dict().items() 
                    if k in ['input_size', 'output_size', 'random_seed', 'normalize_features']})
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.INPUT_SIZE <= 0:
            raise ValueError(f"Input size must be positive, got {self.config.INPUT_SIZE}")
        if self.config.OUTPUT_SIZE <= 0:
            raise ValueError(f"Output size must be positive, got {self.config.OUTPUT_SIZE}")
        if not 0 < self.config.TRAIN_TEST_SPLIT < 1:
            raise ValueError(f"Train-test split must be between 0 and 1, got {self.config.TRAIN_TEST_SPLIT}")
    
    def generate_synthetic_data(self, 
                               n_samples: int = 1000,
                               n_features: Optional[int] = None,
                               n_informative: Optional[int] = None,
                               n_redundant: int = 0,
                               n_repeated: int = 0,
                               noise: float = 0.1,
                               random_state: Optional[int] = None,
                               problem_type: str = 'regression') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data for ADALINE training and evaluation.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features (defaults to config.INPUT_SIZE)
            n_informative: Number of informative features
            n_redundant: Number of redundant features
            n_repeated: Number of repeated features
            noise: Standard deviation of Gaussian noise
            random_state: Random seed for reproducibility
            problem_type: Type of problem ('regression' or 'classification')
            
        Returns:
            Tuple of (X, y) where X is features and y is targets
            
        Raises:
            ValueError: If problem_type is invalid or parameters are invalid
        """
        if problem_type not in ['regression', 'classification']:
            raise ValueError(f"Invalid problem_type: {problem_type}. Must be 'regression' or 'classification'")
        
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        
        n_features = n_features or self.config.INPUT_SIZE
        n_informative = n_informative or min(n_features, 2)
        random_state = random_state or self.config.RANDOM_SEED
        
        logger.info(f"Generating synthetic {problem_type} data: "
                   f"n_samples={n_samples}, n_features={n_features}, "
                   f"n_informative={n_informative}, noise={noise}")
        
        try:
            if problem_type == 'regression':
                X, y = make_regression(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_informative=n_informative,
                    noise=noise,
                    random_state=random_state
                )
                # Ensure y is 2D for consistency
                y = y.reshape(-1, 1)
            else:  # classification
                X, y = make_classification(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_informative=n_informative,
                    n_redundant=n_redundant,
                    n_repeated=n_repeated,
                    n_classes=2,
                    n_clusters_per_class=1,
                    random_state=random_state
                )
                # Convert to regression problem for ADALINE
                y = y.astype(np.float64).reshape(-1, 1)
            
            logger.info(f"Generated data with shapes: X={X.shape}, y={y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            raise
    
    def load_iris_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess Iris dataset for ADALINE training.
        
        Returns:
            Tuple of (X, y) where X is features and y is targets
        """
        try:
            from sklearn.datasets import load_iris
            
            logger.info("Loading Iris dataset")
            iris = load_iris()
            X = iris.data[:, :2]  # Use first two features for 2D visualization
            y = (iris.target == 0).astype(np.float64).reshape(-1, 1)  # Binary classification
            
            logger.info(f"Iris data loaded: X={X.shape}, y={y.shape}")
            return X, y
            
        except ImportError:
            logger.error("scikit-learn not available for loading Iris dataset")
            raise
        except Exception as e:
            logger.error(f"Error loading Iris dataset: {e}")
            raise
    
    def preprocess_data(self, 
                       X: np.ndarray, 
                       y: np.ndarray,
                       fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for ADALINE training.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target values of shape (n_samples, n_outputs)
            fit_scaler: Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            Tuple of (X_processed, y_processed)
            
        Raises:
            ValueError: If input data is invalid
        """
        if X.size == 0:
            raise ValueError(ERROR_MESSAGES['data_empty'])
        
        if X.shape[1] != self.config.INPUT_SIZE:
            raise ValueError(ERROR_MESSAGES['invalid_input_shape'].format(
                X.shape[1], self.config.INPUT_SIZE))
        
        if y.shape[1] != self.config.OUTPUT_SIZE:
            raise ValueError(ERROR_MESSAGES['invalid_target_shape'].format(
                y.shape[1], self.config.OUTPUT_SIZE))
        
        logger.info(f"Preprocessing data: X={X.shape}, y={y.shape}")
        
        X_processed = X.copy()
        y_processed = y.copy()
        
        # Normalize features if configured
        if self.config.NORMALIZE_FEATURES:
            if fit_scaler or self.scaler is None:
                self.scaler = StandardScaler()
                X_processed = self.scaler.fit_transform(X_processed)
                self.is_fitted = True
                logger.info("Fitted StandardScaler on training data")
            else:
                X_processed = self.scaler.transform(X_processed)
                logger.info("Applied fitted StandardScaler to data")
        
        # Add bias term if configured
        if self.config.ADD_BIAS_TERM:
            bias_term = np.ones((X_processed.shape[0], 1))
            X_processed = np.hstack([bias_term, X_processed])
            logger.info("Added bias term to features")
        
        logger.info(f"Preprocessed data shapes: X={X_processed.shape}, y={y_processed.shape}")
        return X_processed, y_processed
    
    def split_data(self, 
                   X: np.ndarray, 
                   y: np.ndarray,
                   test_size: Optional[float] = None,
                   validation_size: Optional[float] = None,
                   random_state: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            X: Input features
            y: Target values
            test_size: Proportion of data for test set
            validation_size: Proportion of data for validation set
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing 'X_train', 'y_train', 'X_val', 'y_val', 
            'X_test', 'y_test' arrays
        """
        test_size = test_size or (1 - self.config.TRAIN_TEST_SPLIT)
        validation_size = validation_size or self.config.VALIDATION_SPLIT
        random_state = random_state or self.config.RANDOM_SEED
        
        logger.info(f"Splitting data: test_size={test_size}, validation_size={validation_size}")
        
        # First split: training+validation vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # Second split: training vs validation
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=None
        )
        
        data_splits = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
        
        logger.info(f"Data split complete: "
                   f"train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")
        
        return data_splits
    
    def load_and_preprocess(self, 
                           data_source: str = 'synthetic',
                           **kwargs) -> Dict[str, np.ndarray]:
        """
        Load and preprocess data in one step.
        
        Args:
            data_source: Source of data ('synthetic', 'iris')
            **kwargs: Additional arguments for data generation
            
        Returns:
            Dictionary containing preprocessed data splits
        """
        logger.info(f"Loading and preprocessing data from source: {data_source}")
        
        # Load data
        if data_source == 'synthetic':
            X, y = self.generate_synthetic_data(**kwargs)
        elif data_source == 'iris':
            X, y = self.load_iris_data()
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y, fit_scaler=True)
        
        # Split data
        data_splits = self.split_data(X_processed, y_processed)
        
        logger.info(SUCCESS_MESSAGES['evaluation_complete'])
        return data_splits
    
    def evaluate_predictions(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           metrics: Optional[Tuple[str, ...]] = None) -> Dict[str, float]:
        """
        Evaluate predictions using multiple metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = metrics or self.config.EVALUATION_METRICS
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
        
        results = {}
        
        for metric in metrics:
            try:
                if metric == 'mse':
                    results[metric] = mean_squared_error(y_true, y_pred)
                elif metric == 'mae':
                    results[metric] = mean_absolute_error(y_true, y_pred)
                elif metric == 'r2_score':
                    results[metric] = r2_score(y_true, y_pred)
                else:
                    logger.warning(f"Unknown metric: {metric}")
                    continue
            except Exception as e:
                logger.error(f"Error computing metric {metric}: {e}")
                results[metric] = np.nan
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def get_data_info(self, data_splits: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.
        
        Args:
            data_splits: Dictionary containing data splits
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'total_samples': sum(data_splits[f'X_{split}'].shape[0] 
                               for split in ['train', 'val', 'test']),
            'n_features': data_splits['X_train'].shape[1],
            'n_outputs': data_splits['y_train'].shape[1],
            'train_samples': data_splits['X_train'].shape[0],
            'val_samples': data_splits['X_val'].shape[0],
            'test_samples': data_splits['X_test'].shape[0],
            'feature_ranges': {
                'min': np.min(data_splits['X_train'], axis=0),
                'max': np.max(data_splits['X_train'], axis=0),
                'mean': np.mean(data_splits['X_train'], axis=0),
                'std': np.std(data_splits['X_train'], axis=0)
            },
            'target_ranges': {
                'min': np.min(data_splits['y_train']),
                'max': np.max(data_splits['y_train']),
                'mean': np.mean(data_splits['y_train']),
                'std': np.std(data_splits['y_train'])
            }
        }
        
        logger.info(f"Dataset info: {info}")
        return info
    
    def save_data_info(self, 
                      data_splits: Dict[str, np.ndarray], 
                      filepath: str) -> None:
        """
        Save dataset information to file.
        
        Args:
            data_splits: Dictionary containing data splits
            filepath: Path to save the information
        """
        import json
        
        info = self.get_data_info(data_splits)
        
        # Convert numpy arrays to lists for JSON serialization
        for key, value in info.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        info[key][subkey] = subvalue.tolist()
            elif isinstance(value, np.ndarray):
                info[key] = value.tolist()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(info, f, indent=2)
            logger.info(f"Data info saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data info: {e}")
            raise


def create_data_loader(config: Any = None) -> ADALINEDataLoader:
    """
    Factory function to create an ADALINE data loader.
    
    Args:
        config: Configuration object
        
    Returns:
        ADALINEDataLoader instance
    """
    return ADALINEDataLoader(config) 