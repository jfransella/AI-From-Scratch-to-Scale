# -*- coding: utf-8 -*-
"""Visualization utility for the Hopfield Network."""

import numpy as np


def display_pattern(pattern, title="Pattern"):
    """
    Displays a flattened bipolar pattern as a 2D grid on the console.

    Args:
        pattern (np.ndarray): A flattened 1D NumPy array with bipolar (-1, 1) values.
        title (str): A title to print above the pattern.
    """
    # Assuming a square pattern, calculate the side length.
    side_length = int(np.sqrt(len(pattern)))
    if side_length * side_length != len(pattern):
        raise ValueError("Pattern length must be a perfect square.")

    # Reshape the pattern into a 2D grid.
    grid = pattern.reshape((side_length, side_length))

    print(f"--- {title} ---")
    for row in grid:
        for pixel in row:
            # Use a block character for 'on' pixels and a space for 'off' pixels.
            char = '█' if pixel == 1 else ' '
            print(char, end=' ')
        print()  # Newline at the end of the row
    print("-" * (side_length * 2 + 3))