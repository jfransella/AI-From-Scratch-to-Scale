# -*- coding: utf-8 -*-
"""Data loading and preprocessing for the Perceptron model.

This module provides functions to load datasets for the Perceptron.
It supports simple CSV files for logic gates and can also download,
filter, and process the MNIST dataset for a more complex task.

"""

import pandas as pd
import numpy as np
from torchvision import datasets
from sklearn.datasets import load_iris
from torchvision.transforms import ToTensor


def load_perceptron_data(file_path):
    """Loads a simple dataset from a CSV file.

    The CSV file is expected to have features in the initial columns and the
    class label in the final column.

    Args:
        file_path (str): The path to the input CSV file.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            - y (np.ndarray): Label vector of shape (n_samples,).
    """
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def load_mnist_data():
    """Loads and prepares the MNIST dataset for binary classification.

    This function downloads the MNIST dataset using torchvision, filters it
    to include only images of digits 0 and 1, flattens the images into vectors,
    normalizes pixel values to the [0, 1] range, and returns them as NumPy arrays.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): Training features of shape (n_samples, 784).
            - y (np.ndarray): Binary labels (0 or 1) of shape (n_samples,).
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
    X /= 255.0

    return X, y


def load_iris_data(class_indices, feature_indices=(0, 1)):
    """Loads and prepares a binary classification task from the Iris dataset.

    This function filters the Iris dataset to include only two specified classes
    and two specified features, making it suitable for binary classification and
    2D visualization.

    Args:
        class_indices (list[int]): A list of two integer class indices to use.
                                   (e.g., [0, 1] for Setosa vs Versicolour).
        feature_indices (tuple[int]): A tuple of two feature indices to use
                                      (default: (0, 1) for sepal length/width).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): Filtered features of shape (n_samples, 2).
            - y (np.ndarray): Binary labels (0 or 1) of shape (n_samples,).
                              The first class index is mapped to 0, the second to 1.
    """
    iris = load_iris()
    
    # Filter data for the specified classes
    mask = np.isin(iris.target, class_indices)
    X = iris.data[mask]
    y = iris.target[mask]

    # Select specified features
    X = X[:, list(feature_indices)]

    # Remap the chosen class labels to 0 and 1
    unique_labels = np.unique(y)
    y_binary = np.where(y == unique_labels[0], 0, 1)

    return X, y_binary