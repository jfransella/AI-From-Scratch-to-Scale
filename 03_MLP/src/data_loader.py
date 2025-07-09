# -*- coding: utf-8 -*-
"""Data loading and preprocessing for the MLP model."""

import logging
from typing import Tuple
import pandas as pd
import numpy as np
from scipy.ndimage import shift
from torchvision import datasets
from torchvision.transforms import ToTensor

logger = logging.getLogger(__name__)


def _shift_image(image: np.ndarray, max_shift: int = 4) -> np.ndarray:
    """
    Shifts a 784-element flattened image by a random amount.

    Args:
        image: A 784-element numpy array representing the image
        max_shift: The maximum number of pixels to shift in any direction

    Returns:
        The shifted 784-element flattened image
        
    Raises:
        ValueError: If image is not 784 elements long
    """
    if image.size != 784:
        raise ValueError(f"Expected image with 784 elements, got {image.size}")
    
    # Reshape the flattened image back to 28x28 before shifting
    image_2d = image.reshape(28, 28)
    shift_x = np.random.randint(-max_shift, max_shift + 1)
    shift_y = np.random.randint(-max_shift, max_shift + 1)
    # Shift the 2D image. `cval=0` fills new pixels with black.
    shifted_image_2d = shift(image_2d, [shift_y, shift_x], cval=0, mode='constant')
    # Flatten the image back to a 784-element vector
    return shifted_image_2d.flatten()


def load_logic_gate_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a simple dataset for a logic gate from a CSV file.

    The CSV file is expected to have features in the initial columns and the
    class label in the final column.

    Args:
        file_path: The path to the input CSV file

    Returns:
        A tuple containing the feature matrix (X) and label vector (y)
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV file is empty or malformed
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f"CSV file {file_path} is empty")
        
        logger.info(f"Loaded logic gate data from {file_path}: {df.shape[0]} samples")
        
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)  # Ensure y is a column vector
        return X, y
    except FileNotFoundError:
        logger.error(f"Logic gate data file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading logic gate data from {file_path}: {e}")
        raise ValueError(f"Failed to load logic gate data: {e}")


def load_mnist_multiclass_data(return_test_set: bool = False) -> Tuple[np.ndarray, ...]:
    """Loads and prepares the full MNIST dataset for multi-class classification.

    This function downloads the MNIST dataset, flattens the images into
    vectors, normalizes pixel values, and one-hot encodes the labels for both
    the training and (optionally) test sets.

    Args:
        return_test_set: Whether to return test set in addition to training set

    Returns:
        A tuple containing:
        - X_train: Training features of shape (60000, 784)
        - y_train: One-hot encoded training labels of shape (60000, 10)
        - X_test: Test features of shape (10000, 784) if requested
        - y_test: One-hot encoded test labels of shape (10000, 10) if requested
        
    Raises:
        RuntimeError: If MNIST data cannot be downloaded or processed
    """
    try:
        # Helper function to process a dataset
        def _process_dataset(dataset):
            X = dataset.data.numpy()
            y_labels = dataset.targets.numpy()
            n_samples = X.shape[0]
            X = X.reshape(n_samples, -1).astype(np.float32)
            X /= 255.0
            n_classes = 10
            y_one_hot = np.eye(n_classes, dtype=np.float32)[y_labels]
            return X, y_one_hot

        # Load and process training data
        logger.info("Loading MNIST training data...")
        training_data = datasets.MNIST(
            root="data", train=True, download=True, transform=ToTensor()
        )
        X_train, y_train = _process_dataset(training_data)
        logger.info(f"Loaded MNIST training data: {X_train.shape[0]} samples")

        if not return_test_set:
            return X_train, y_train

        # Load and process test data
        logger.info("Loading MNIST test data...")
        test_data = datasets.MNIST(
            root="data", train=False, download=True, transform=ToTensor()
        )
        X_test, y_test = _process_dataset(test_data)
        logger.info(f"Loaded MNIST test data: {X_test.shape[0]} samples")

        return X_train, y_train, X_test, y_test
    
    except Exception as e:
        logger.error(f"Error loading MNIST data: {e}")
        raise RuntimeError(f"Failed to load MNIST data: {e}")


def load_mnist_failure_test_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset and creates a modified test set where images
    are randomly shifted.

    This is designed to test the model's robustness to translations.

    Returns:
        A tuple containing:
        - X_train: Original training features
        - y_train: Original training labels
        - X_test_shifted: Test features with each image shifted
        - y_test: Original test labels
        
    Raises:
        RuntimeError: If MNIST data cannot be loaded or processed
    """
    try:
        logger.info("Loading MNIST data for failure test (with shifted images)...")
        # Load the standard MNIST data first, including the test set
        X_train, y_train, X_test, y_test = load_mnist_multiclass_data(return_test_set=True)

        # Apply the random shift to each image in the test set
        logger.info("Applying random shifts to test images...")
        X_test_shifted = np.array([_shift_image(img) for img in X_test], dtype=np.float32)
        logger.info(f"Created shifted test set with {X_test_shifted.shape[0]} images")

        return X_train, y_train, X_test_shifted, y_test
    
    except Exception as e:
        logger.error(f"Error creating failure test data: {e}")
        raise RuntimeError(f"Failed to create failure test data: {e}")