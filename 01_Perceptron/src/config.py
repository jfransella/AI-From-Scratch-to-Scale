# -*- coding: utf-8 -*-
"""Configuration settings for the Perceptron model.

This file centralizes all hyperparameters, file paths, and other
configuration values for the Perceptron project. Storing configuration
in a separate file makes it easier to manage and modify experimental
parameters without changing the core logic of the application.

Attributes:
    LEARNING_RATE (float): The step size for each weight update during training.
                           A smaller value leads to slower but potentially more
                           stable convergence.
    EPOCHS (int): The total number of times the training algorithm will iterate
                  over the entire dataset.
    INPUT_DATA_PATH (str): The file path for the input dataset. This is where
                           the data_loader will look for the training data.

"""

LEARNING_RATE = 0.1
EPOCHS = 50
INPUT_DATA_PATH = "data/perceptron_data.csv"