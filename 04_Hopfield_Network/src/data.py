# -*- coding: utf-8 -*-
"""Data definitions for the Hopfield Network."""

import numpy as np

# Define patterns as 2D lists of 0s and 1s for readability.
# 1 represents an "on" pixel, 0 represents an "off" pixel.
_patterns_map = {
    'C': [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ],
    'L': [
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ],
    'I': [
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ],
    'T': [
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ],
    'X': [
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1],
    ],
    'O': [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ],
}

def get_patterns():
    """
    Returns a dictionary of named patterns for the Hopfield network.

    The patterns are 5x5 ASCII art letters, flattened into 25-element
    bipolar (-1, 1) vectors.

    Returns:
        dict[str, np.ndarray]: A dictionary where keys are pattern names
                               (e.g., 'C') and values are the corresponding
                               bipolar NumPy arrays.
    """
    patterns = {}
    for name, pattern_2d in _patterns_map.items():
        # Flatten the 2D list and convert to a NumPy array
        flat_pattern = np.array(pattern_2d).flatten()
        # Convert from (0, 1) to bipolar (-1, 1) which is required by the model
        bipolar_pattern = np.where(flat_pattern == 0, -1, 1)
        patterns[name] = bipolar_pattern
    return patterns