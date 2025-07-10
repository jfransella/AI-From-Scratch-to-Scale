"""
Pattern Generation and Data Loading for Hopfield Network
=======================================================

This module creates and manages binary patterns for associative memory experiments.
Unlike supervised learning datasets, Hopfield networks work with patterns that serve
as both input and target (auto-associative memory).

Educational Focus:
- Pattern design for memory experiments
- Noise corruption for robustness testing  
- Visualization of binary patterns
- Understanding pattern orthogonality and capacity limits

Mathematical Context:
- Binary patterns: {-1, +1}^N
- Pattern orthogonality affects storage capacity
- Noise corruption: Hamming distance from original pattern
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path

try:
    # Try relative imports first (when run as module)
    from .config import (
        PATTERN_HEIGHT, PATTERN_WIDTH, PATTERN_SIZE, PATTERN_ON, PATTERN_OFF,
        PATTERN_TYPES, NOISE_LEVELS, DEFAULT_NOISE_LEVEL, MAX_PATTERNS,
        RANDOM_SEED, USE_FIXED_SEED, PLOTS_DIR, DATA_DIR
    )
except ImportError:
    # Fall back to absolute imports (when run as script)
    from config import (
        PATTERN_HEIGHT, PATTERN_WIDTH, PATTERN_SIZE, PATTERN_ON, PATTERN_OFF,
        PATTERN_TYPES, NOISE_LEVELS, DEFAULT_NOISE_LEVEL, MAX_PATTERNS,
        RANDOM_SEED, USE_FIXED_SEED, PLOTS_DIR, DATA_DIR
    )

try:
    # Try relative imports first (when run as module)
    from .visualize import visualize_pattern, visualize_pattern_set
except ImportError:
    # Fall back to absolute imports (when run as script)
    from visualize import visualize_pattern, visualize_pattern_set

# Set up logging
logger = logging.getLogger(__name__)

class PatternGenerator:
    """
    Generates binary patterns for Hopfield Network experiments.
    
    This class focuses on creating educational patterns that demonstrate
    key concepts in associative memory and energy-based learning.
    """
    
    def __init__(self, height: int = PATTERN_HEIGHT, width: int = PATTERN_WIDTH):
        """
        Initialize pattern generator.
        
        Args:
            height: Pattern grid height
            width: Pattern grid width
        """
        self.height = height
        self.width = width
        self.size = height * width
        
        if USE_FIXED_SEED:
            np.random.seed(RANDOM_SEED)
        
        logger.info(f"PatternGenerator initialized: {height}x{width} = {self.size} neurons")
    
    def generate_simple_shapes(self) -> Dict[str, np.ndarray]:
        """
        Generate basic geometric patterns for educational demonstrations.
        
        Returns:
            Dictionary mapping shape names to binary patterns
            
        Note:
            These patterns are designed to be visually interpretable
            and have good separation for robust retrieval.
        """
        patterns = {}
        
        # Create blank pattern template
        def blank_pattern():
            return np.full((self.height, self.width), PATTERN_OFF, dtype=int)
        
        # Cross pattern
        cross = blank_pattern()
        mid_h, mid_w = self.height // 2, self.width // 2
        cross[mid_h, :] = PATTERN_ON  # Horizontal line
        cross[:, mid_w] = PATTERN_ON  # Vertical line
        patterns['cross'] = cross.flatten()
        
        # Circle pattern (approximate)
        circle = blank_pattern()
        center_h, center_w = self.height // 2, self.width // 2
        radius = min(self.height, self.width) // 3
        for i in range(self.height):
            for j in range(self.width):
                dist = np.sqrt((i - center_h)**2 + (j - center_w)**2)
                if abs(dist - radius) < 1.0:  # Circle boundary
                    circle[i, j] = PATTERN_ON
        patterns['circle'] = circle.flatten()
        
        # Square pattern
        square = blank_pattern()
        margin = 2
        square[margin:-margin, margin] = PATTERN_ON      # Left edge
        square[margin:-margin, -margin-1] = PATTERN_ON   # Right edge  
        square[margin, margin:-margin] = PATTERN_ON      # Top edge
        square[-margin-1, margin:-margin] = PATTERN_ON   # Bottom edge
        patterns['square'] = square.flatten()
        
        # Triangle pattern (improved)
        triangle = blank_pattern()
        mid_w = self.width // 2
        base_y = self.height - 3  # Bottom of triangle
        top_y = 2  # Top of triangle
        
        # Create triangle by drawing lines from top to base
        for y in range(top_y, base_y + 1):
            # Calculate width of triangle at this height
            if base_y - top_y == 0:
                # Degenerate case: single row triangle
                progress = 0
            else:
                progress = (y - top_y) / (base_y - top_y)  # 0 at top, 1 at base
            half_width = int(progress * (self.width // 3))  # Triangle width grows
            
            # Draw the triangle outline (left and right edges)
            if half_width == 0:
                # Top point
                triangle[y, mid_w] = PATTERN_ON
            else:
                # Left and right edges
                left_x = mid_w - half_width
                right_x = mid_w + half_width
                if left_x >= 0:
                    triangle[y, left_x] = PATTERN_ON
                if right_x < self.width:
                    triangle[y, right_x] = PATTERN_ON
        
        # Draw the base line
        base_start = max(0, mid_w - (self.width // 3))
        base_end = min(self.width, mid_w + (self.width // 3) + 1)
        triangle[base_y, base_start:base_end] = PATTERN_ON
        patterns['triangle'] = triangle.flatten()
        
        logger.info(f"Generated {len(patterns)} simple shape patterns")
        return patterns
    
    def generate_letter_patterns(self) -> Dict[str, np.ndarray]:
        """
        Generate letter patterns for character recognition experiments.
        
        Returns:
            Dictionary mapping letters to binary patterns
        """
        patterns = {}
        
        def blank_pattern():
            return np.full((self.height, self.width), PATTERN_OFF, dtype=int)
        
        # Letter A
        A = blank_pattern()
        mid_h = self.height // 2
        mid_w = self.width // 2
        # Vertical lines
        A[2:-1, mid_w-2] = PATTERN_ON
        A[2:-1, mid_w+2] = PATTERN_ON
        # Top and middle horizontal
        A[2, mid_w-1:mid_w+2] = PATTERN_ON
        A[mid_h, mid_w-1:mid_w+2] = PATTERN_ON
        patterns['A'] = A.flatten()
        
        # Letter B (simplified)
        B = blank_pattern()
        # Vertical line
        B[1:-1, 2] = PATTERN_ON
        # Horizontal lines
        B[1, 2:6] = PATTERN_ON
        B[mid_h, 2:6] = PATTERN_ON
        B[-2, 2:6] = PATTERN_ON
        # Right edges
        B[2:mid_h, 5] = PATTERN_ON
        B[mid_h+1:-2, 5] = PATTERN_ON
        patterns['B'] = B.flatten()
        
        # Add more letters as needed...
        
        logger.info(f"Generated {len(patterns)} letter patterns")
        return patterns
    
    def get_educational_letters(self) -> Dict[str, np.ndarray]:
        """
        Get the classic educational letter patterns (C, L, I, T, X, O).
        
        These are the original 5x5 patterns used in basic Hopfield demonstrations,
        originally defined in data.py. Moved here for centralized pattern management.
        
        Returns:
            Dictionary mapping letter names to bipolar patterns
            
        Educational Focus:
            Simple, recognizable patterns perfect for understanding
            basic associative memory concepts and pattern completion.
        """
        # Define patterns as 2D lists of 0s and 1s for readability
        _classic_patterns = {
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
        
        patterns = {}
        for name, pattern_2d in _classic_patterns.items():
            # Flatten the 2D list and convert to a NumPy array
            flat_pattern = np.array(pattern_2d).flatten()
            # Convert from (0, 1) to bipolar (-1, 1) which is required by the model
            bipolar_pattern = np.where(flat_pattern == 0, PATTERN_OFF, PATTERN_ON)
            patterns[name] = bipolar_pattern
            
        logger.info(f"Generated {len(patterns)} classic educational letter patterns")
        return patterns
    
    def generate_random_patterns(self, num_patterns: int, density: float = 0.5) -> List[np.ndarray]:
        """
        Generate random binary patterns.
        
        Args:
            num_patterns: Number of patterns to generate
            density: Probability of a neuron being 'on' (default: 0.5 for balanced)
            
        Returns:
            List of random binary patterns
            
        Note:
            Random patterns are useful for studying capacity limits and
            testing the theoretical predictions of Hopfield networks.
        """
        patterns = []
        
        for i in range(num_patterns):
            # Generate random binary pattern
            random_pattern = np.random.choice(
                [PATTERN_OFF, PATTERN_ON], 
                size=self.size, 
                p=[1-density, density]
            )
            patterns.append(random_pattern)
        
        logger.info(f"Generated {num_patterns} random patterns with density {density}")
        return patterns
    
    def add_noise(self, pattern: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add noise to a pattern by flipping random bits.
        
        Args:
            pattern: Original binary pattern
            noise_level: Fraction of bits to flip (0.0 to 1.0)
            
        Returns:
            Noisy version of the pattern
            
        Mathematical Note:
            Noise adds energy to the pattern, moving it away from the
            stored attractor. The network's job is to recover the original.
        """
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError(f"Noise level must be between 0 and 1, got {noise_level}")
        
        noisy_pattern = pattern.copy()
        num_flips = int(noise_level * len(pattern))
        
        if num_flips > 0:
            # Randomly select positions to flip
            flip_positions = np.random.choice(len(pattern), num_flips, replace=False)
            # Flip the selected bits
            noisy_pattern[flip_positions] *= -1
        
        return noisy_pattern
    
    def calculate_overlap(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """
        Calculate overlap between two patterns.
        
        Args:
            pattern1: First binary pattern
            pattern2: Second binary pattern
            
        Returns:
            Overlap value between -1 and 1
            
        Mathematical Definition:
            overlap = (1/N) * Σ(s1_i * s2_i)
            where N is the pattern size and s_i ∈ {-1, +1}
        """
        if len(pattern1) != len(pattern2):
            raise ValueError("Patterns must have same length")
        
        overlap = np.mean(pattern1 * pattern2)
        return overlap
    
    def create_simple_digit(self, digit_type: str, size: int = 10) -> np.ndarray:
        """
        Create simple digit-like patterns for educational demonstrations.
        
        Args:
            digit_type: Type of digit ('cross', 'square', 'L', 'T', 'plus')
            size: Size of the pattern grid
            
        Returns:
            Binary pattern as flattened array
        """
        pattern = np.full((size, size), PATTERN_OFF, dtype=np.float32)
        center = size // 2
        
        if digit_type == 'cross':
            # Create a cross pattern
            pattern[center, :] = PATTERN_ON  # Horizontal line
            pattern[:, center] = PATTERN_ON  # Vertical line
            
        elif digit_type == 'square':
            # Create a square border
            margin = 2
            pattern[margin:size-margin, margin] = PATTERN_ON  # Left edge
            pattern[margin:size-margin, size-margin-1] = PATTERN_ON  # Right edge
            pattern[margin, margin:size-margin] = PATTERN_ON  # Top edge
            pattern[size-margin-1, margin:size-margin] = PATTERN_ON  # Bottom edge
            
        elif digit_type == 'L':
            # Create an L shape
            pattern[2:size-1, 2] = PATTERN_ON  # Vertical line
            pattern[size-2, 2:size-2] = PATTERN_ON  # Horizontal line
            
        elif digit_type == 'T':
            # Create a T shape
            pattern[2, 2:size-2] = PATTERN_ON  # Top horizontal line
            pattern[2:size-2, center] = PATTERN_ON  # Vertical line
            
        elif digit_type == 'plus':
            # Create a smaller plus sign
            start, end = center-2, center+3
            pattern[center, start:end] = PATTERN_ON  # Horizontal
            pattern[start:end, center] = PATTERN_ON  # Vertical
            
        else:
            raise ValueError(f"Unknown digit type: {digit_type}")
        
        return pattern.flatten()


class HopfieldDataLoader:
    """
    Main data loader class for Hopfield Network experiments.
    
    This class orchestrates pattern generation, noise addition, and
    data management for various experimental scenarios.
    """
    
    def __init__(self):
        """Initialize the Hopfield data loader."""
        self.generator = PatternGenerator()
        self.stored_patterns = {}
        self.pattern_history = []
        
        # Create output directories
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
        
        logger.info("HopfieldDataLoader initialized")
    
    def load_pattern_set(self, pattern_type: str) -> Dict[str, np.ndarray]:
        """
        Load a specific set of patterns.
        
        Args:
            pattern_type: Type of patterns to load ('simple_shapes', 'letters', etc.)
            
        Returns:
            Dictionary of pattern_name -> pattern_array
        """
        if pattern_type == "simple_shapes":
            patterns = self.generator.generate_simple_shapes()
        elif pattern_type == "letters":
            patterns = self.generator.generate_letter_patterns()
        elif pattern_type == "random":
            # Generate random patterns
            random_patterns = self.generator.generate_random_patterns(MAX_PATTERNS)
            patterns = {f"random_{i}": pattern for i, pattern in enumerate(random_patterns)}
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        self.stored_patterns.update(patterns)
        logger.info(f"Loaded {len(patterns)} patterns of type '{pattern_type}'")
        
        return patterns
    
    def create_noisy_test_set(self, patterns: Dict[str, np.ndarray], 
                             noise_levels: List[float] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Create noisy versions of patterns for retrieval testing.
        
        Args:
            patterns: Original patterns to add noise to
            noise_levels: List of noise levels to apply
            
        Returns:
            Nested dictionary: {pattern_name: {noise_level: noisy_pattern}}
        """
        if noise_levels is None:
            noise_levels = NOISE_LEVELS
        
        noisy_patterns = {}
        
        for pattern_name, pattern in patterns.items():
            noisy_patterns[pattern_name] = {}
            
            for noise_level in noise_levels:
                noisy_pattern = self.generator.add_noise(pattern, noise_level)
                noisy_patterns[pattern_name][f"noise_{noise_level:.1f}"] = noisy_pattern
        
        logger.info(f"Created noisy test set with {len(noise_levels)} noise levels")
        return noisy_patterns
    
    def analyze_pattern_statistics(self, patterns: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze statistical properties of pattern set.
        
        Args:
            patterns: Dictionary of patterns to analyze
            
        Returns:
            Dictionary of statistical measures
        """
        pattern_list = list(patterns.values())
        
        # Convert to numpy array for analysis
        pattern_matrix = np.array(pattern_list)  # Shape: (num_patterns, pattern_size)
        
        # Calculate statistics
        stats = {
            'num_patterns': len(patterns),
            'pattern_size': len(pattern_list[0]),
            'mean_activity': np.mean(pattern_matrix == PATTERN_ON),
            'pattern_balance': np.mean(np.sum(pattern_matrix, axis=1)),  # Balance around 0
        }
        
        # Pairwise overlaps
        overlaps = []
        for i in range(len(pattern_list)):
            for j in range(i+1, len(pattern_list)):
                overlap = self.generator.calculate_overlap(pattern_list[i], pattern_list[j])
                overlaps.append(abs(overlap))
        
        stats['mean_abs_overlap'] = np.mean(overlaps) if overlaps else 0.0
        stats['max_abs_overlap'] = np.max(overlaps) if overlaps else 0.0
        
        logger.info(f"Pattern statistics: {stats}")
        return stats
    
    def save_patterns(self, patterns: Dict[str, np.ndarray], filename: str) -> None:
        """
        Save patterns to file for later use.
        
        Args:
            patterns: Dictionary of patterns to save
            filename: Name of file to save to
        """
        filepath = Path(DATA_DIR) / filename
        np.savez(filepath, **patterns)
        logger.info(f"Saved {len(patterns)} patterns to {filepath}")
    
    def load_patterns(self, filename: str) -> Dict[str, np.ndarray]:
        """
        Load patterns from file.
        
        Args:
            filename: Name of file to load from
            
        Returns:
            Dictionary of loaded patterns
        """
        filepath = Path(DATA_DIR) / filename
        loaded_data = np.load(filepath)
        patterns = {key: loaded_data[key] for key in loaded_data.keys()}
        
        logger.info(f"Loaded {len(patterns)} patterns from {filepath}")
        return patterns
    
    def get_experiment_data(self, experiment_type: str) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Get data prepared for specific experiment type.
        
        Args:
            experiment_type: Type of experiment ('capacity', 'noise', 'retrieval')
            
        Returns:
            Tuple of (patterns, experiment_config)
        """
        if experiment_type == "capacity":
            # Use random patterns for capacity experiments
            patterns = self.load_pattern_set("random")
            config = {"focus": "storage_capacity", "measure": "retrieval_success"}
            
        elif experiment_type == "noise_robustness":
            # Use simple shapes for noise robustness
            patterns = self.load_pattern_set("simple_shapes")
            config = {"focus": "noise_tolerance", "measure": "recovery_rate"}
            
        elif experiment_type == "retrieval":
            # Use letters for retrieval demonstration
            patterns = self.load_pattern_set("letters")
            config = {"focus": "associative_retrieval", "measure": "convergence_steps"}
            
        else:
            # Default: simple shapes
            patterns = self.load_pattern_set("simple_shapes")
            config = {"focus": "basic_demonstration", "measure": "general"}
        
        return patterns, config


def demonstrate_data_loading():
    """
    Demonstrate the data loading and pattern generation capabilities.
    
    This function serves as both a test and an educational example
    of the different types of patterns available for Hopfield networks.
    """
    print("="*80)
    print("HOPFIELD NETWORK DATA LOADING DEMONSTRATION")
    print("="*80)
    
    # Initialize data loader
    loader = HopfieldDataLoader()
    
    # Load different pattern types
    print("\n1. Loading Simple Shapes...")
    shapes = loader.load_pattern_set("simple_shapes")
    loader.generator.visualize_pattern_set(shapes, "Simple Shape Patterns")
    
    print("\n2. Analyzing Pattern Statistics...")
    stats = loader.analyze_pattern_statistics(shapes)
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n3. Demonstrating Noise Addition...")
    cross_pattern = shapes['cross']
    noise_levels = [0.0, 0.2, 0.4, 0.6]
    
    noisy_examples = {}
    for noise in noise_levels:
        noisy_pattern = loader.generator.add_noise(cross_pattern, noise)
        overlap = loader.generator.calculate_overlap(cross_pattern, noisy_pattern)
        noisy_examples[f"noise_{noise:.1f}_overlap_{overlap:.2f}"] = noisy_pattern
    
    loader.generator.visualize_pattern_set(noisy_examples, "Noise Corruption Examples")
    
    print("\n4. Creating Experimental Dataset...")
    patterns, config = loader.get_experiment_data("noise_robustness")
    print(f"Experiment focus: {config['focus']}")
    print(f"Primary measure: {config['measure']}")
    print(f"Number of patterns: {len(patterns)}")
    
    print("\nData loading demonstration complete!")
    print("="*80)


def create_simple_digit(digit_type: str, size: int = 10) -> np.ndarray:
    """
    Convenience function to create simple digit patterns.
    
    Args:
        digit_type: Type of digit ('cross', 'square', 'L', 'T', 'plus')
        size: Size of the pattern grid
        
    Returns:
        Binary pattern as flattened array
    """
    generator = PatternGenerator(height=size, width=size)
    return generator.create_simple_digit(digit_type, size)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demonstration
    demonstrate_data_loading()
