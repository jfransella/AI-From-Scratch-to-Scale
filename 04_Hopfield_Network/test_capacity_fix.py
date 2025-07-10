#!/usr/bin/env python3
"""
Quick test script to verify capacity experiment fix
"""

import sys
import os
sys.path.append('src')

from data_loader import HopfieldDataLoader
from model import HopfieldNetwork
import numpy as np

def test_capacity_fix():
    """Test the capacity experiment with corrected threshold"""
    print("Testing Hopfield Network Capacity with 60% threshold...")
    
    # Initialize components
    loader = HopfieldDataLoader()
    
    # Test different pattern counts
    pattern_counts = [5, 10, 15]
    
    for num_patterns in pattern_counts:
        print(f"\nTesting {num_patterns} patterns:")
        
        trial_successes = []
        
        # Run 3 trials
        for trial in range(3):
            # Generate random patterns with 20% density
            patterns = loader.generator.generate_random_patterns(num_patterns, density=0.2)
            
            # Create fresh network and store patterns
            network = HopfieldNetwork(100)
            network.store_patterns(patterns)
            
            # Test retrieval with noise
            successful_retrievals = 0
            
            for pattern in patterns:
                # Add 20% noise
                noisy = loader.generator.add_noise(pattern, 0.2)
                
                # Attempt retrieval
                retrieved, info = network.retrieve_pattern(noisy)
                
                # Check success with 60% threshold
                if info['best_overlap'] >= 0.6:
                    successful_retrievals += 1
            
            success_rate = successful_retrievals / num_patterns
            trial_successes.append(success_rate)
            print(f"  Trial {trial+1}: {success_rate:.1%} success")
        
        avg_success = np.mean(trial_successes)
        print(f"  Average: {avg_success:.1%} success rate")
        
        # Expected: should see decline as patterns increase
        # 5 patterns: ~80-100% (well below capacity)
        # 10 patterns: ~60-80% (approaching capacity) 
        # 15 patterns: ~30-60% (at/above capacity)

if __name__ == "__main__":
    test_capacity_fix()
