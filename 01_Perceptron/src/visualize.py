# -*- coding: utf-8 -*-
"""Visualization functions for the Perceptron model.

This module provides functions to generate and save plots that help visualize
the Perceptron's performance, such as the decision boundary.

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import logging


def plot_decision_boundary(X, y, model, filename="decision_boundary.png"):
    """Plots the decision boundary of a trained Perceptron model.

    Args:
        X (np.ndarray): The input features.
        y (np.ndarray): The true labels.
        model (Perceptron): The trained Perceptron instance.
        filename (str): The name of the file to save the plot.
    """
    fig, ax = plt.subplots()

    # Scatter plot for the data points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o',
               edgecolor='k', label='Data points')

    # Create the decision boundary line
    x0_min, x0_max = ax.get_xlim()
    
    # Formula for the decision line: w1*x1 + w2*x2 + b = 0
    # => x2 = (-w1*x1 - b) / w2
    w = model.weights
    b = model.bias
    
    # Ensure we don't divide by zero if a weight is zero
    if w[1] != 0:
        x1_vals = np.array([x0_min, x0_max])
        x2_vals = -(w[0] * x1_vals + b) / w[1]
        ax.plot(x1_vals, x2_vals, 'g--', label='Decision Boundary')

    ax.set_title("Perceptron Decision Boundary")
    ax.set_xlabel("Input Feature 1")
    ax.set_ylabel("Input Feature 2")
    ax.legend()
    ax.grid(True)

    # Save the plot
    output_dir = "outputs/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close(fig)
    
    # Use logger from the training script to report saving
    logging.info(f"Decision boundary plot saved to {save_path}")
    
def plot_perceptron_weights(model, filename="perceptron_weights.png"):
    """Visualizes the Perceptron's weights as a 28x28 image.

    This is useful for image classification tasks to see what features
    the model has learned.

    Args:
        model (Perceptron): The trained Perceptron instance.
        filename (str): The name of the file to save the plot.
    """
    if model.weights is None or model.weights.shape[0] != 784:
        logging.warning("Weight visualization is only applicable for 28x28 image data (784 features).")
        return

    fig, ax = plt.subplots()
    # Reshape the 784-element weight vector into a 28x28 image
    weights_img = np.reshape(model.weights, (28, 28))

    # Use imshow to display the weights as a heatmap
    im = ax.imshow(weights_img, cmap=plt.cm.coolwarm)
    fig.colorbar(im, ax=ax)

    ax.set_title("Perceptron Learned Weights")
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the plot
    output_dir = "outputs/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close(fig)

    logging.info(f"Model weights visualization saved to {save_path}")