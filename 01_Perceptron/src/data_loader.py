# -*- coding: utf-8 -*-
"""Data loading and preprocessing for the Perceptron model.

This module provides a function to load the dataset for the Perceptron
from a specified CSV file. It handles reading the data, separating it
into features and labels, and returning them in a format suitable for
model training (NumPy arrays).

"""

import pandas as pd


def load_perceptron_data(file_path):
    """Loads the Perceptron dataset from a CSV file.

    Args:
        file_path (str): The path to the input CSV file. The file is
                         expected to have feature columns and a final
                         'label' column.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): The feature matrix.
            - y (np.ndarray): The label vector.
    """
    df = pd.read_csv(file_path)

    # Assumes the last column is the label and the rest are features.
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y