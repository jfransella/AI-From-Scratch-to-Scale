"""
Spatial Invariance Demonstration for Hopfield Networks
=====================================================

This module demonstrates WHY Hopfield networks are not spatially invariant,
providing important educational context for understanding the limitations
of early neural network architectures and the motivation for CNNs.

Educational Objectives:
- Show that Hopfield networks are position-specific
- Demonstrate the failure of shifted pattern retrieval
- Explain why this limitation led to convolutional architectures
- Provide intuition for the curse of dimensionality in memory models
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import logging
from pathlib import Path
import os

try:
    # Try relative imports first (when run as module)
    from .model import HopfieldNetwork
    from .data_loader import create_simple_digit
    from .config import PLOTS_DIR
except ImportError:
    # Fall back to absolute imports (when run as script)
    from model import HopfieldNetwork
    from data_loader import create_simple_digit
    from config import PLOTS_DIR

logger = logging.getLogger(__name__)


def create_shifted_patterns(pattern: np.ndarray, shifts: List[Tuple[int, int]]) -> List[np.ndarray]:
    """
    Create shifted versions of a pattern to test spatial invariance.
    
    Args:
        pattern: Original binary pattern (flattened)
        shifts: List of (row_shift, col_shift) tuples
        
    Returns:
        List of shifted patterns
    """
    # Reshape to 2D for shifting
    size = int(np.sqrt(len(pattern)))
    pattern_2d = pattern.reshape(size, size)
    
    shifted_patterns = []
    
    for row_shift, col_shift in shifts:
        # Create shifted pattern with zero padding
        shifted = np.zeros_like(pattern_2d)
        
        # Calculate valid regions
        src_row_start = max(0, -row_shift)
        src_row_end = min(size, size - row_shift)
        src_col_start = max(0, -col_shift)
        src_col_end = min(size, size - col_shift)
        
        dst_row_start = max(0, row_shift)
        dst_row_end = dst_row_start + (src_row_end - src_row_start)
        dst_col_start = max(0, col_shift)
        dst_col_end = dst_col_start + (src_col_end - src_col_start)
        
        # Copy valid region
        if src_row_end > src_row_start and src_col_end > src_col_start:
            shifted[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = \
                pattern_2d[src_row_start:src_row_end, src_col_start:src_col_end]
        
        shifted_patterns.append(shifted.flatten())
    
    return shifted_patterns


def demonstrate_spatial_limitation():
    """
    Demonstrate that Hopfield networks cannot handle spatial translations.
    
    This is an important educational demonstration showing why position-specific
    associative memory is insufficient for real-world pattern recognition.
    """
    logger.info("Running spatial invariance limitation demonstration...")
    
    # Create network
    network = HopfieldNetwork()
    
    # Create a simple digit pattern (e.g., a cross shape)
    original_pattern = create_simple_digit('cross', size=int(np.sqrt(network.size)))
    
    # Store the original pattern
    network.store_patterns([original_pattern])
    logger.info(f"Stored original pattern at center position")
    
    # Test shifts
    shifts = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (-1, 0), (0, -1)]
    shift_labels = ['Original', 'Down 1', 'Right 1', 'Down+Right 1', 
                   'Down 2', 'Right 2', 'Up 1', 'Left 1']
    
    # Create shifted versions
    shifted_patterns = create_shifted_patterns(original_pattern, shifts)
    
    # Test retrieval for each shifted pattern
    results = []
    overlaps = []
    
    for i, (shifted_pattern, label) in enumerate(zip(shifted_patterns, shift_labels)):
        # Retrieve using shifted pattern as input
        retrieved, _, overlap = network.retrieve_pattern(
            shifted_pattern, return_overlap=True
        )
        
        # Calculate overlap with original stored pattern
        final_overlap = np.mean(retrieved == original_pattern)
        
        results.append({
            'shift': shifts[i],
            'label': label,
            'input_pattern': shifted_pattern,
            'retrieved_pattern': retrieved,
            'overlap_with_original': final_overlap,
            'retrieval_successful': final_overlap > 0.8
        })
        
        overlaps.append(final_overlap)
        
        logger.info(f"{label}: overlap = {final_overlap:.3f}, "
                   f"successful = {final_overlap > 0.8}")
    
    # Create visualization
    create_spatial_invariance_plot(results, shift_labels, overlaps)
    
    # Print educational summary
    print_educational_summary(results)
    
    return results


def create_spatial_invariance_plot(results: List[Dict], labels: List[str], overlaps: List[float]):
    """Create visualization showing spatial invariance failure."""
    
    n_patterns = len(results)
    size = int(np.sqrt(len(results[0]['input_pattern'])))
    
    # Create subplot grid
    fig, axes = plt.subplots(3, n_patterns, figsize=(2*n_patterns, 6))
    fig.suptitle('Hopfield Network Spatial Invariance Limitation', fontsize=16)
    
    for i, (result, label) in enumerate(zip(results, labels)):
        # Input pattern
        axes[0, i].imshow(result['input_pattern'].reshape(size, size), 
                         cmap='RdBu', vmin=-1, vmax=1)
        axes[0, i].set_title(f'Input: {label}')
        axes[0, i].axis('off')
        
        # Retrieved pattern
        axes[1, i].imshow(result['retrieved_pattern'].reshape(size, size), 
                         cmap='RdBu', vmin=-1, vmax=1)
        axes[1, i].set_title(f'Retrieved')
        axes[1, i].axis('off')
        
        # Overlap score
        color = 'green' if result['retrieval_successful'] else 'red'
        axes[2, i].bar(0, result['overlap_with_original'], color=color, alpha=0.7)
        axes[2, i].set_ylim(0, 1)
        axes[2, i].set_title(f'Overlap: {result["overlap_with_original"]:.2f}')
        axes[2, i].set_xticks([])
        axes[2, i].axhline(y=0.8, color='black', linestyle='--', alpha=0.5)
    
    # Add row labels
    axes[0, 0].set_ylabel('Input Pattern', fontsize=12)
    axes[1, 0].set_ylabel('Retrieved Pattern', fontsize=12)
    axes[2, 0].set_ylabel('Overlap Score', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'spatial_invariance_limitation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = ['green' if overlap > 0.8 else 'red' for overlap in overlaps]
    bars = ax.bar(range(len(labels)), overlaps, color=colors, alpha=0.7)
    
    ax.set_xlabel('Shift Type')
    ax.set_ylabel('Overlap with Original Pattern')
    ax.set_title('Spatial Invariance Test Results\n(Red = Failed Retrieval, Green = Successful)')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.axhline(y=0.8, color='black', linestyle='--', alpha=0.5, label='Success Threshold')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'spatial_invariance_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Spatial invariance plots saved")


def print_educational_summary(results: List[Dict]):
    """Print educational summary of spatial invariance demonstration."""
    
    print("\n" + "="*70)
    print("EDUCATIONAL SUMMARY: Hopfield Network Spatial Invariance Limitation")
    print("="*70)
    
    successful = sum(1 for r in results if r['retrieval_successful'])
    total = len(results)
    
    print(f"\nRESULTS:")
    print(f"- Successful retrievals: {successful}/{total}")
    print(f"- Success rate: {successful/total*100:.1f}%")
    
    print(f"\nKEY INSIGHTS:")
    print(f"1. POSITION DEPENDENCE: Hopfield networks store patterns at specific positions")
    print(f"2. NO SPATIAL INVARIANCE: Shifting by even 1 pixel can cause retrieval failure")
    print(f"3. MEMORY SPECIFICITY: Each weight w_ij connects specific pixel positions")
    print(f"4. HISTORICAL MOTIVATION: This limitation led to convolutional architectures")
    
    print(f"\nWHY THIS HAPPENS:")
    print(f"- Hopfield weights: w_ij = correlation between positions i and j")
    print(f"- Shifting changes which positions are correlated")
    print(f"- Network has no mechanism to recognize 'same pattern, different location'")
    
    print(f"\nWHAT THIS TEACHES:")
    print(f"- CNNs solve this with weight sharing and translation equivariance")
    print(f"- Hopfield networks are best for fixed-position pattern matching")
    print(f"- Understanding limitations helps appreciate modern architectures")
    
    print(f"\nPRACTICAL IMPLICATIONS:")
    print(f"- Hopfield networks: Good for error correction, content-addressable memory")
    print(f"- Hopfield networks: Poor for image recognition requiring spatial invariance")
    print(f"- Modern applications: Optimization, memory models, attention mechanisms")
    
    print("="*70)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demonstration
    results = demonstrate_spatial_limitation()
    
    print(f"\nDemonstration complete! Check {PLOTS_DIR} for visualizations.")
