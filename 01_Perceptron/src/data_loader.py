# -*- coding: utf-8 -*-
"""Data loading and preprocessing for the Perceptron model.

This module provides functions to load datasets for the Perceptron.
It supports simple CSV files for logic gates and can also download,
filter, and process the MNIST dataset for a more complex task.

"""

import pandas as pd
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor


def load_perceptron_data(file_path):
    """Loads a simple dataset from a CSV file.

    Args:
        file_path (str): The path to the input CSV file. Assumes the
                         last column is the 'label'.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the feature
                                       matrix (X) and label vector (y).
    """
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def load_mnist_data():
    """Loads and prepares the MNIST dataset for 0s and 1s.

    This function downloads the MNIST dataset using torchvision, filters it
    to include only images of digits 0 and 1, flattens the images into
    vectors, and returns them as NumPy arrays.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the training
                                       features (X) and labels (y).
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