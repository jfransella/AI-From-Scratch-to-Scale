"""
Test the new Hopfield visualizer with shared framework
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.visualize_new import HopfieldVisualizer, display_pattern

def test_hopfield_visualizer():
    """Test the HopfieldVisualizer functionality."""
    print("Testing HopfieldVisualizer with shared framework...")
    
    # Create test patterns
    np.random.seed(42)
    size = 5  # 5x5 patterns for testing
    
    # Create some simple test patterns
    pattern1 = np.array([1, 1, 1, -1, -1,
                        1, -1, 1, -1, -1,
                        1, 1, 1, -1, -1,
                        1, -1, 1, -1, -1,
                        1, -1, 1, -1, -1])
    
    pattern2 = np.array([-1, 1, 1, 1, -1,
                        -1, -1, 1, -1, -1,
                        -1, -1, 1, -1, -1,
                        -1, -1, 1, -1, -1,
                        -1, 1, 1, 1, -1])
    
    patterns = {"Letter L": pattern1, "Letter I": pattern2}
    
    # Test console display (backwards compatibility)
    print("\n--- Testing console display ---")
    display_pattern(pattern1, "Test Pattern L")
    
    # Test visualizer
    print("\n--- Testing HopfieldVisualizer ---")
    viz = HopfieldVisualizer(default_save_dir=Path("test_outputs"))
    
    try:
        # Test single pattern visualization
        print("✓ Testing single pattern visualization...")
        fig, ax = viz.visualize_pattern(pattern1, "Test Pattern L", show=False)
        
        # Test pattern set visualization
        print("✓ Testing pattern set visualization...")
        fig2, axes2 = viz.visualize_pattern_set(patterns, "Test Pattern Set", show=False)
        
        # Test energy landscape (create dummy weights)
        print("✓ Testing energy landscape...")
        weights = np.random.random((25, 25)) * 0.1
        fig3, ax3 = viz.visualize_energy_landscape([pattern1, pattern2], weights, show=False)
        
        # Test convergence analysis
        print("✓ Testing convergence analysis...")
        convergence_data = {
            'energy_histories': [[1.0, 0.8, 0.6, 0.4, 0.2], [1.2, 0.9, 0.5, 0.3, 0.1]],
            'convergence_steps': [5, 4, 6, 3, 5, 4],
            'energy_decreases': [0.8, 0.9, 0.7, 1.0, 0.8, 0.6],
            'initial_overlaps': [0.6, 0.7, 0.5, 0.8, 0.6, 0.7],
            'final_overlaps': [0.9, 0.95, 0.85, 0.98, 0.92, 0.88]
        }
        fig4, axes4 = viz.plot_convergence_analysis(convergence_data, show=False)
        
        print("✅ All HopfieldVisualizer tests passed!")
        
        # Clean up
        viz.cleanup_figures()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Create test output directory
    Path("test_outputs").mkdir(exist_ok=True)
    
    success = test_hopfield_visualizer()
    if success:
        print("\n🎉 Hopfield visualizer integration successful!")
    else:
        print("\n💥 Hopfield visualizer integration failed!")
