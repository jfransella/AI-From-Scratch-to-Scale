# -*- coding: utf-8 -*-
"""Main script to demonstrate the Hopfield Network."""

import numpy as np

from src.model import HopfieldNetwork
from src.data import get_patterns
from src.visualize import display_pattern


def corrupt_pattern(pattern, num_flips=5):
    """
    Flips a specified number of bits in a pattern.

    Args:
        pattern (np.ndarray): The original pattern.
        num_flips (int): The number of bits to flip.

    Returns:
        np.ndarray: The corrupted pattern.
    """
    corrupted = np.copy(pattern)
    # Choose random indices to flip without replacement
    flip_indices = np.random.choice(len(pattern), size=num_flips, replace=False)
    # Flip the bits at the chosen indices (-1 becomes 1, and 1 becomes -1)
    corrupted[flip_indices] *= -1
    return corrupted


def main():
    """
    Main function to run the Hopfield Network demonstration.
    """
    # 1. Load patterns
    patterns_dict = get_patterns()
    all_patterns_list = list(patterns_dict.values())
    all_patterns_names = list(patterns_dict.keys())
    num_neurons = len(all_patterns_list[0])

    # --- Original Demonstration ---
    print("--- Single Pattern Recall Demonstration ---")
    network = HopfieldNetwork(num_neurons=num_neurons)
    # Train on the first 3 patterns
    network.train(all_patterns_list[:3])
    print(f"Stored 3 patterns: {all_patterns_names[:3]}")
    original_pattern_name = 'C'
    original_pattern = patterns_dict[original_pattern_name]
    display_pattern(original_pattern, title=f"Original Pattern '{original_pattern_name}'")
    corrupted = corrupt_pattern(original_pattern, num_flips=5)
    display_pattern(corrupted, title="Corrupted Pattern")
    recalled_pattern = network.predict(corrupted)
    display_pattern(recalled_pattern, title="Recalled Pattern")
    print("\n" + "="*40 + "\n")

    # --- Capacity Test Demonstration ---
    print("--- Network Capacity Test ---")
    original_pattern_to_test = patterns_dict['C']
    corrupted_to_test = corrupt_pattern(original_pattern_to_test, num_flips=5)

    # We will test the network's ability to recall 'C' as we add more patterns.
    for num_stored in range(1, len(all_patterns_list) + 1):
        print(f"\n--- Testing with {num_stored} stored patterns ---")
        patterns_to_store = all_patterns_list[:num_stored]
        pattern_names = all_patterns_names[:num_stored]

        network = HopfieldNetwork(num_neurons=num_neurons)
        network.train(patterns_to_store)
        print(f"Storing: {pattern_names}")

        # Attempt to recall the corrupted 'C'
        recalled = network.predict(corrupted_to_test)

        if np.array_equal(recalled, original_pattern_to_test):
            print("Recall successful!")
            display_pattern(recalled, title="Recalled 'C' Correctly")
        else:
            print("Recall FAILED.")
            display_pattern(recalled, title="Incorrectly Recalled Pattern")


if __name__ == "__main__":
    main()