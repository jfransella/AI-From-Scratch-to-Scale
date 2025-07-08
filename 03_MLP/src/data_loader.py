# -*- coding: utf-8 -*-
"""Data loading and preprocessing for the MLP model."""

import pandas as pd
import numpy as np
from scipy.ndimage import shift
from torchvision import datasets
from torchvision.transforms import ToTensor


def _shift_image(image, max_shift=4):
    """
    Shifts a 784-element flattened image by a random amount.

    Args:
        image (np.ndarray): A 784-element numpy array representing the image.
        max_shift (int): The maximum number of pixels to shift in any direction.

    Returns:
        np.ndarray: The shifted 784-element flattened image.
    """
    # Reshape the flattened image back to 28x28 before shifting
    image_2d = image.reshape(28, 28)
    shift_x = np.random.randint(-max_shift, max_shift + 1)
    shift_y = np.random.randint(-max_shift, max_shift + 1)
    # Shift the 2D image. `cval=0` fills new pixels with black.
    shifted_image_2d = shift(image_2d, [shift_y, shift_x], cval=0, mode='constant')
    # Flatten the image back to a 784-element vector
    return shifted_image_2d.flatten()


def load_logic_gate_data(file_path):
    """Loads a simple dataset for a logic gate from a CSV file.

    The CSV file is expected to have features in the initial columns and the
    class label in the final column.

    Args:
        file_path (str): The path to the input CSV file.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the feature
                                       matrix (X) and label vector (y).
    """
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)  # Ensure y is a column vector
    return X, y


def load_mnist_multiclass_data(return_test_set=False):
    """Loads and prepares the full MNIST dataset for multi-class classification.

    This function downloads the MNIST dataset, flattens the images into
    vectors, normalizes pixel values, and one-hot encodes the labels for both
    the training and (optionally) test sets.

    Returns:
        tuple[np.ndarray, ...]: A tuple containing:
            - X_train (np.ndarray): Training features of shape (60000, 784).
            - y_train (np.ndarray): One-hot encoded training labels of shape (60000, 10).
            - X_test (np.ndarray): Test features of shape (10000, 784) if requested.
            - y_test (np.ndarray): One-hot encoded test labels of shape (10000, 10) if requested.
    """
    # Helper function to process a dataset
    def _process_dataset(dataset):
        X = dataset.data.numpy()
        y_labels = dataset.targets.numpy()
        n_samples = X.shape[0]
        X = X.reshape(n_samples, -1).astype('float32')
        X /= 255.0
        n_classes = 10
        y_one_hot = np.eye(n_classes)[y_labels]
        return X, y_one_hot

    # Load and process training data
    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    X_train, y_train = _process_dataset(training_data)

    if not return_test_set:
        return X_train, y_train

    # Load and process test data
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )
    X_test, y_test = _process_dataset(test_data)

    return X_train, y_train, X_test, y_test


def load_mnist_failure_test_data():
    """
    Loads the MNIST dataset and creates a modified test set where images
    are randomly shifted.

    This is designed to test the model's robustness to translations.

    Returns:
        tuple[np.ndarray, ...]: A tuple containing:
            - X_train (np.ndarray): Original training features.
            - y_train (np.ndarray): Original training labels.
            - X_test_shifted (np.ndarray): Test features with each image shifted.
            - y_test (np.ndarray): Original test labels.
    """
    # Load the standard MNIST data first, including the test set
    X_train, y_train, X_test, y_test = load_mnist_multiclass_data(return_test_set=True)

    # Apply the random shift to each image in the test set
    X_test_shifted = np.array([_shift_image(img) for img in X_test])

    return X_train, y_train, X_test_shifted, y_test