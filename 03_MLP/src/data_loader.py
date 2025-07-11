# -*- coding: utf-8 -*-
"""Data loading and preprocessing for the MLP model.

This module handles all data loading and preprocessing operations, including:
- Logic gate data from CSV files
- MNIST dataset loading and preprocessing  
- Data augmentation for robustness testing

Educational Context:
    Data preprocessing is a critical step in machine learning pipelines.
    This module demonstrates proper data validation, normalization techniques,
    and robustness testing through data augmentation.
"""

import logging
import os
from typing import Tuple
import pandas as pd
import numpy as np
from scipy.ndimage import shift
from torchvision import datasets
from torchvision.transforms import ToTensor

logger = logging.getLogger(__name__)

# Constants for data processing
MNIST_IMAGE_HEIGHT: int = 28
MNIST_IMAGE_WIDTH: int = 28
MNIST_PIXEL_MAX: float = 255.0
DEFAULT_MAX_SHIFT: int = 4


def _shift_image(image: np.ndarray, max_shift: int = DEFAULT_MAX_SHIFT) -> np.ndarray:
    """Shifts a 784-element flattened MNIST image by a random amount.
    
    Educational Context:
        This function simulates real-world variations in image positioning.
        By randomly shifting images, we can test model robustness to translation
        invariance - a crucial property for practical computer vision systems.

    Args:
        image: A 784-element numpy array representing the flattened 28x28 image
        max_shift: The maximum number of pixels to shift in any direction

    Returns:
        The shifted 784-element flattened image
        
    Raises:
        ValueError: If image is not exactly 784 elements (28x28 flattened)
    """
    expected_size = MNIST_IMAGE_HEIGHT * MNIST_IMAGE_WIDTH
    if image.size != expected_size:
        raise ValueError(f"Expected image with {expected_size} elements (28x28 flattened), got {image.size}")
    
    # Reshape the flattened image back to 28x28 for spatial operations
    image_2d = image.reshape(MNIST_IMAGE_HEIGHT, MNIST_IMAGE_WIDTH)
    
    # Generate random shifts in both x and y directions
    shift_x = np.random.randint(-max_shift, max_shift + 1)
    shift_y = np.random.randint(-max_shift, max_shift + 1)
    
    # Apply the shift using scipy.ndimage.shift
    # cval=0 fills new pixels with black (background)
    # mode='constant' ensures we don't wrap pixels around
    shifted_image_2d = shift(image_2d, [shift_y, shift_x], cval=0, mode='constant')
    
    # Ensure the shifted image maintains proper normalization [0, 1]
    # Clip any interpolation artifacts that might go outside this range
    shifted_image_2d = np.clip(shifted_image_2d, 0, 1)
    
    # Return flattened image to maintain consistent interface
    return shifted_image_2d.flatten().astype(np.float32)


def load_logic_gate_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a simple dataset for a logic gate from a CSV file.
    
    Educational Context:
        Logic gates (AND, OR, XOR, NAND) provide simple, well-understood problems
        for testing neural network implementations. XOR is particularly important
        as it's the simplest non-linearly separable problem.

    The CSV file is expected to have features in the initial columns and the
    class label in the final column.

    Args:
        file_path: The path to the input CSV file

    Returns:
        A tuple containing:
        - X: Feature matrix of shape (n_samples, n_features)
        - y: Label vector of shape (n_samples, 1)
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV file is empty or malformed
    """
    # Validate file exists before attempting to load
    if not os.path.exists(file_path):
        logger.error(f"Logic gate data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        logger.info(f"Loading logic gate data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Validate data is not empty
        if df.empty:
            raise ValueError(f"CSV file {file_path} is empty")
        
        # Validate minimum structure (at least 2 columns: 1 feature + 1 label)
        if df.shape[1] < 2:
            raise ValueError(f"CSV file must have at least 2 columns, got {df.shape[1]}")
        
        logger.info(f"Loaded logic gate data: {df.shape[0]} samples, {df.shape[1]-1} features")
        
        # Split features and labels
        # All columns except the last are features
        X = df.iloc[:, :-1].values.astype(np.float32)
        # Last column is the label, reshaped to column vector
        y = df.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)
        
        # Validate data ranges for typical logic gate problems
        if np.any((X < 0) | (X > 1)):
            logger.warning("Logic gate features contain values outside [0,1] range")
        if np.any((y < 0) | (y > 1)):
            logger.warning("Logic gate labels contain values outside [0,1] range")
        
        return X, y
        
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file {file_path} is empty or corrupted")
        raise ValueError(f"Failed to load logic gate data: file is empty or corrupted")
    except Exception as e:
        logger.error(f"Error loading logic gate data from {file_path}: {e}")
        raise ValueError(f"Failed to load logic gate data: {e}")


def load_mnist_multiclass_data(return_test_set: bool = False) -> Tuple[np.ndarray, ...]:
    """Loads and prepares the full MNIST dataset for multi-class classification.
    
    Educational Context:
        MNIST preprocessing demonstrates key concepts in computer vision:
        1. Pixel normalization (0-255 → 0-1) for numerical stability
        2. Flattening 2D images into 1D vectors for fully connected networks
        3. One-hot encoding for multi-class classification
        4. Train/test split for proper evaluation

    This function downloads the MNIST dataset, flattens the images into
    vectors, normalizes pixel values, and one-hot encodes the labels for both
    the training and (optionally) test sets.

    Args:
        return_test_set: Whether to return test set in addition to training set

    Returns:
        A tuple containing:
        - X_train: Training features of shape (60000, 784), normalized to [0,1]
        - y_train: One-hot encoded training labels of shape (60000, 10)
        - X_test: Test features of shape (10000, 784) if requested
        - y_test: One-hot encoded test labels of shape (10000, 10) if requested
        
    Raises:
        RuntimeError: If MNIST data cannot be downloaded or processed
    """
    try:
        def _process_dataset(dataset):
            """Helper function to process MNIST dataset consistently."""
            # Extract numpy arrays from PyTorch tensors
            X = dataset.data.numpy()
            y_labels = dataset.targets.numpy()
            n_samples = X.shape[0]
            
            # Flatten 28x28 images into 784-element vectors
            # Each pixel becomes a feature
            X = X.reshape(n_samples, -1).astype(np.float32)
            
            # Normalize pixel values from [0, 255] to [0, 1]
            # This improves numerical stability and convergence
            X = X / MNIST_PIXEL_MAX
            
            # Convert class labels to one-hot encoding
            # This is required for multi-class classification with softmax + cross-entropy
            n_classes = 10  # MNIST has digits 0-9
            y_one_hot = np.eye(n_classes, dtype=np.float32)[y_labels]
            
            return X, y_one_hot

        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)

        # Load and process training data
        logger.info("Loading MNIST training data...")
        training_data = datasets.MNIST(
            root="data", 
            train=True, 
            download=True, 
            transform=ToTensor()
        )
        X_train, y_train = _process_dataset(training_data)
        logger.info(f"Loaded MNIST training data: {X_train.shape[0]} samples, "
                   f"image shape: {X_train.shape[1]} features")

        # Validate training data
        assert X_train.shape == (60000, 784), f"Unexpected training data shape: {X_train.shape}"
        assert y_train.shape == (60000, 10), f"Unexpected training label shape: {y_train.shape}"
        assert X_train.min() >= 0 and X_train.max() <= 1, "Training data not properly normalized"

        if not return_test_set:
            return X_train, y_train

        # Load and process test data
        logger.info("Loading MNIST test data...")
        test_data = datasets.MNIST(
            root="data", 
            train=False, 
            download=True, 
            transform=ToTensor()
        )
        X_test, y_test = _process_dataset(test_data)
        logger.info(f"Loaded MNIST test data: {X_test.shape[0]} samples")

        # Validate test data
        assert X_test.shape == (10000, 784), f"Unexpected test data shape: {X_test.shape}"
        assert y_test.shape == (10000, 10), f"Unexpected test label shape: {y_test.shape}"
        assert X_test.min() >= 0 and X_test.max() <= 1, "Test data not properly normalized"

        return X_train, y_train, X_test, y_test
    
    except Exception as e:
        logger.error(f"Error loading MNIST data: {e}")
        raise RuntimeError(f"Failed to load MNIST data: {e}")


def load_mnist_failure_test_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads the MNIST dataset and creates a modified test set where images are randomly shifted.
    
    Educational Context:
        Robustness testing is crucial for evaluating real-world model performance.
        This function creates a challenging test set by applying random translations
        to MNIST images, simulating the kind of variations a model might encounter
        in practice. The performance difference between original and shifted images
        measures translation invariance.

    This is designed to test the model's robustness to translations, which is
    an important property for practical computer vision systems.

    Returns:
        A tuple containing:
        - X_train: Original training features of shape (60000, 784)
        - y_train: Original training labels of shape (60000, 10)
        - X_test_shifted: Test features with each image randomly shifted (10000, 784)
        - y_test: Original test labels of shape (10000, 10)
        
    Raises:
        RuntimeError: If MNIST data cannot be loaded or processed
    """
    try:
        logger.info("Loading MNIST data for robustness test (with shifted images)...")
        
        # Load the standard MNIST data first, including the test set
        X_train, y_train, X_test, y_test = load_mnist_multiclass_data(return_test_set=True)

        # Apply random shifts to each image in the test set
        logger.info("Applying random shifts to test images for robustness evaluation...")
        
        # Set random seed for reproducible shift patterns
        np.random.seed(42)
        
        # Apply shifts to all test images
        X_test_shifted = np.array([
            _shift_image(img, max_shift=DEFAULT_MAX_SHIFT) 
            for img in X_test
        ], dtype=np.float32)
        
        logger.info(f"Created shifted test set with {X_test_shifted.shape[0]} images")
        logger.info(f"Each image shifted by up to {DEFAULT_MAX_SHIFT} pixels in each direction")

        # Validate shifted data maintains proper range
        assert X_test_shifted.min() >= 0 and X_test_shifted.max() <= 1, \
            "Shifted data not properly normalized"
        assert X_test_shifted.shape == X_test.shape, \
            "Shifted data shape doesn't match original"

        return X_train, y_train, X_test_shifted, y_test
    
    except Exception as e:
        logger.error(f"Error creating failure test data: {e}")
        raise RuntimeError(f"Failed to create failure test data: {e}")