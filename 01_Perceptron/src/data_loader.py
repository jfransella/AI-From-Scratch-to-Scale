# -*- coding: utf-8 -*-
"""Data loading and preprocessing for the Perceptron model.

This module provides functions to load datasets for the Perceptron.
It supports simple CSV files for logic gates and can also download,
filter, and process the MNIST dataset for a more complex task.

"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from torchvision import datasets
from sklearn.datasets import load_iris
from torchvision.transforms import ToTensor

# Handle both relative and absolute imports
try:
    from .constants import PIXEL_NORMALIZATION_FACTOR
except ImportError:
    from constants import PIXEL_NORMALIZATION_FACTOR


def load_perceptron_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a simple dataset from a CSV file.

    The CSV file is expected to have features in the initial columns and the
    class label in the final column.

    Args:
        file_path: The path to the input CSV file.

    Returns:
        A tuple containing:
            - X: Feature matrix of shape (n_samples, n_features).
            - y: Label vector of shape (n_samples,).
            
    Raises:
        FileNotFoundError: If the specified file doesn't exist.
        ValueError: If the file is empty or malformed.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    if df.empty:
        raise ValueError(f"CSV file is empty: {file_path}")
    if df.shape[1] < 2:
        raise ValueError(f"CSV file must have at least 2 columns (features + label), got {df.shape[1]}")
    
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int32)
    return X, y

def load_mnist_data() -> Tuple[np.ndarray, np.ndarray]:
    """Loads and prepares the MNIST dataset for binary classification.

    This function downloads the MNIST dataset using torchvision, filters it
    to include only images of digits 0 and 1, flattens the images into vectors,
    normalizes pixel values to the [0, 1] range, and returns them as NumPy arrays.

    Returns:
        A tuple containing:
            - X: Training features of shape (n_samples, 784).
            - y: Binary labels (0 or 1) of shape (n_samples,).
    """
    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Filter for digits 0 and 1
    idx = (training_data.targets == 0) | (training_data.targets == 1)
    X = training_data.data[idx].numpy()
    y = training_data.targets[idx].numpy()

    # Flatten the images from (N, 28, 28) to (N, 784)
    n_samples = X.shape[0]
    X = X.reshape(n_samples, -1).astype('float32')

    # Normalize pixel values to be between 0 and 1
    X = X.astype(np.float32) / PIXEL_NORMALIZATION_FACTOR

    return X, y


def load_iris_data(class_indices: List[int], feature_indices: Tuple[int, int] = (0, 1)) -> Tuple[np.ndarray, np.ndarray]:
    """Loads and prepares a binary classification task from the Iris dataset.

    This function filters the Iris dataset to include only two specified classes
    and two specified features, making it suitable for binary classification and
    2D visualization.

    Args:
        class_indices: A list of two integer class indices to use.
                      (e.g., [0, 1] for Setosa vs Versicolour).
        feature_indices: A tuple of two feature indices to use
                        (default: (0, 1) for sepal length/width).

    Returns:
        A tuple containing:
            - X: Filtered features of shape (n_samples, 2).
            - y: Binary labels (0 or 1) of shape (n_samples,).
                 The first class index is mapped to 0, the second to 1.
                 
    Raises:
        ValueError: If class_indices or feature_indices are invalid.
    """
    if len(class_indices) != 2:
        raise ValueError(f"class_indices must contain exactly 2 classes, got {len(class_indices)}")
    if len(feature_indices) != 2:
        raise ValueError(f"feature_indices must contain exactly 2 features, got {len(feature_indices)}")
    
    iris = load_iris()
    
    # Validate class indices
    if not all(0 <= idx < len(iris.target_names) for idx in class_indices):
        raise ValueError(f"Invalid class indices: {class_indices}. Must be in range [0, {len(iris.target_names)-1}]")
    
    # Validate feature indices  
    if not all(0 <= idx < iris.data.shape[1] for idx in feature_indices):
        raise ValueError(f"Invalid feature indices: {feature_indices}. Must be in range [0, {iris.data.shape[1]-1}]")
    
    # Filter data for the specified classes
    mask = np.isin(iris.target, class_indices)
    X = iris.data[mask]
    y = iris.target[mask]

    # Select specified features
    X = X[:, list(feature_indices)].astype(np.float32)

    # Remap the chosen class labels to 0 and 1
    unique_labels = np.unique(y)
    y_binary = np.where(y == unique_labels[0], 0, 1).astype(np.int32)

    return X, y_binary