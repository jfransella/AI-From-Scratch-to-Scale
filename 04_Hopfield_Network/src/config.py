"""
Configuration for Hopfield Network Energy-Based Associative Memory
================================================================

This configuration follows the educational philosophy of the AI-From-Scratch-to-Scale project,
focusing on understanding energy-based models as an alternative paradigm to supervised learning.

Mathematical Foundation:
- Energy Function: E = -0.5 * Σ(i,j) w_ij * s_i * s_j - Σ(i) θ_i * s_i
- Lyapunov Function: Energy always decreases during updates
- Symmetric Weights: w_ij = w_ji (ensures convergence)
- Pattern Storage: Hebbian-like rule for storing patterns

Historical Context:
John Hopfield (1982) introduced this model connecting neural computation to statistical mechanics,
showing how neural networks could function as content-addressable memory systems.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

# ============================================================================
# HOPFIELD NETWORK ARCHITECTURE PARAMETERS
# ============================================================================

# Network Size - Educational Focus on Small Networks
NETWORK_SIZE: int = 100  # Number of neurons (keep small for visualization)
MIN_NETWORK_SIZE: int = 16  # Minimum for pattern visualization
MAX_NETWORK_SIZE: int = 400  # Maximum for computational feasibility

# ============================================================================
# PATTERN STORAGE AND MEMORY CAPACITY
# ============================================================================

# Pattern Dimensions
PATTERN_HEIGHT: int = 10  # Height of pattern grid (10x10 = 100 neurons)
PATTERN_WIDTH: int = 10   # Width of pattern grid
PATTERN_SIZE: int = PATTERN_HEIGHT * PATTERN_WIDTH

# Memory Capacity (Rule of thumb: ~0.15 * N patterns for good retrieval)
MAX_PATTERNS: int = 15   # Maximum patterns to store
CAPACITY_RATIO: float = 0.15  # Theoretical capacity as fraction of network size

# Pattern Values
PATTERN_ON: int = 1      # Active state
PATTERN_OFF: int = -1    # Inactive state

# ============================================================================
# ENERGY-BASED DYNAMICS PARAMETERS
# ============================================================================

# Update Rules
UPDATE_RULE: str = "asynchronous"  # Options: "asynchronous", "synchronous"
MAX_ITERATIONS: int = 1000         # Maximum iterations for convergence
CONVERGENCE_THRESHOLD: float = 1e-6  # Energy change threshold for convergence

# Temperature for Probabilistic Updates (optional)
TEMPERATURE: float = 0.0  # T=0 for deterministic, T>0 for stochastic
USE_TEMPERATURE: bool = False

# ============================================================================
# PATTERN GENERATION AND NOISE PARAMETERS
# ============================================================================

# Pattern Types for Experiments
PATTERN_TYPES = {
    "simple_shapes": ["cross", "circle", "square", "triangle"],
    "letters": ["A", "B", "C", "D", "E"],
    "digits": ["0", "1", "2", "3", "4"],
    "random": "random_binary"
}

# Noise Settings for Pattern Corruption
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Fraction of bits to flip
DEFAULT_NOISE_LEVEL: float = 0.2

# ============================================================================
# TRAINING (PATTERN STORAGE) PARAMETERS
# ============================================================================

# Hebbian Learning Rule
LEARNING_RULE: str = "hebbian"  # Options: "hebbian", "delta", "pseudoinverse"
NORMALIZE_WEIGHTS: bool = False  # Whether to normalize weight matrix

# Weight Constraints
ZERO_DIAGONAL: bool = True  # Set w_ii = 0 (no self-connections)
SYMMETRIC_WEIGHTS: bool = True  # Enforce w_ij = w_ji

# ============================================================================
# EVALUATION AND ANALYSIS PARAMETERS
# ============================================================================

# Retrieval Metrics
OVERLAP_THRESHOLD: float = 0.80  # Minimum overlap for successful retrieval (80% is realistic)
MAX_RETRIEVAL_STEPS: int = 100   # Maximum steps for pattern retrieval

# Energy Landscape Analysis
ENERGY_SAMPLE_SIZE: int = 1000   # Number of random states for energy sampling
BASIN_ANALYSIS: bool = True      # Whether to analyze attractor basins

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Pattern Display
PATTERN_FIGSIZE: Tuple[int, int] = (12, 8)
ENERGY_FIGSIZE: Tuple[int, int] = (10, 6)
CONVERGENCE_FIGSIZE: Tuple[int, int] = (10, 6)

# Color Schemes
PATTERN_COLORMAP: str = "RdBu"    # For pattern visualization
ENERGY_COLORMAP: str = "viridis"  # For energy landscape
GRID_COLOR: str = "gray"
ACTIVE_COLOR: str = "black"
INACTIVE_COLOR: str = "white"

# Animation Settings
ANIMATION_INTERVAL: int = 200     # Milliseconds between frames
SAVE_ANIMATIONS: bool = False     # Whether to save energy evolution animations

# ============================================================================
# EXPERIMENTAL CONFIGURATIONS
# ============================================================================

# Capacity Experiments
CAPACITY_EXPERIMENT_SIZES = [10, 25, 50, 100, 200]
CAPACITY_TRIALS: int = 10  # Number of trials per configuration

# Noise Robustness Experiments
NOISE_EXPERIMENT_LEVELS = np.linspace(0.0, 0.8, 17)  # 0% to 80% noise
NOISE_TRIALS: int = 20

# Convergence Analysis
CONVERGENCE_TRIALS: int = 50
CONVERGENCE_TIMEOUT: int = 500

# ============================================================================
# COMPARISON EXPERIMENTS (PARADIGM CONTRAST)
# ============================================================================

# Comparison with Supervised Learning
COMPARISON_MODELS = ["perceptron", "mlp"]
COMPARISON_DATASETS = ["pattern_completion", "denoising", "classification"]

# Performance Metrics for Comparison
COMPARISON_METRICS = [
    "pattern_capacity",
    "noise_robustness", 
    "convergence_speed",
    "energy_minimization"
]

# ============================================================================
# PHYSICS AND MATHEMATICAL CONNECTIONS
# ============================================================================

# Statistical Mechanics Analogies
SPIN_GLASS_ANALOGY: bool = True   # Explain connections to spin glasses
ISING_MODEL_COMPARISON: bool = True  # Compare with Ising model

# Energy Function Components
INCLUDE_BIAS_TERMS: bool = False  # Whether to include threshold terms
BIAS_LEARNING_RATE: float = 0.01

# ============================================================================
# EDUCATIONAL FEATURES
# ============================================================================

# Mathematical Documentation
DETAILED_DERIVATIONS: bool = True  # Include step-by-step math
PHYSICS_CONNECTIONS: bool = True   # Explain physics analogies
HISTORICAL_CONTEXT: bool = True    # Include historical background

# Interactive Features
INTERACTIVE_PATTERNS: bool = True  # Allow manual pattern creation
STEP_BY_STEP_MODE: bool = True    # Show each iteration
ENERGY_TRACKING: bool = True      # Track energy at each step

# ============================================================================
# OUTPUT AND LOGGING CONFIGURATION
# ============================================================================

# Directory Structure
OUTPUT_DIR: str = "outputs"
MODELS_DIR: str = f"{OUTPUT_DIR}/models"
PLOTS_DIR: str = f"{OUTPUT_DIR}/plots"
LOGS_DIR: str = f"{OUTPUT_DIR}/logs"
DATA_DIR: str = f"{OUTPUT_DIR}/data"

# Logging Configuration
LOG_LEVEL: str = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE: bool = True
LOG_TO_CONSOLE: bool = True

# Model Saving
SAVE_WEIGHTS: bool = True
SAVE_PATTERNS: bool = True
SAVE_ENERGY_HISTORY: bool = True

# ============================================================================
# WEIGHTS & BIASES EXPERIMENT TRACKING
# ============================================================================

# Project Configuration
WANDB_PROJECT_NAME: str = "hopfield-network-education"
WANDB_ENTITY: Optional[str] = None  # Will use default user entity

# Experiment Tracking
LOG_VISUALIZATIONS: bool = True  # Log plots and visualizations to W&B
LOG_METRICS_FREQUENCY: int = 1   # Log metrics every N experiments
SAVE_MODEL_ARTIFACTS: bool = True  # Save model weights as W&B artifacts

# ============================================================================
# DEVELOPMENT AND DEBUGGING
# ============================================================================

# Debug Settings
DEBUG_MODE: bool = False
VERBOSE_OUTPUT: bool = True
CHECK_ENERGY_MONOTONIC: bool = True  # Verify energy always decreases
VALIDATE_CONVERGENCE: bool = True

# Reproducibility
RANDOM_SEED: int = 42
USE_FIXED_SEED: bool = True

# ============================================================================
# HELPER FUNCTIONS FOR CONFIGURATION VALIDATION
# ============================================================================

def validate_config() -> Dict[str, Any]:
    """Validate configuration parameters and return summary."""
    
    issues = []
    warnings = []
    
    # Check network size constraints
    if NETWORK_SIZE != PATTERN_SIZE:
        issues.append(f"Network size ({NETWORK_SIZE}) must equal pattern size ({PATTERN_SIZE})")
    
    # Check capacity constraints
    theoretical_capacity = int(CAPACITY_RATIO * NETWORK_SIZE)
    if MAX_PATTERNS > theoretical_capacity:
        warnings.append(f"Max patterns ({MAX_PATTERNS}) exceeds theoretical capacity ({theoretical_capacity})")
    
    # Check pattern dimensions
    if PATTERN_HEIGHT * PATTERN_WIDTH != PATTERN_SIZE:
        issues.append("Pattern dimensions inconsistent with pattern size")
    
    # Check update rule validity
    valid_updates = ["asynchronous", "synchronous"]
    if UPDATE_RULE not in valid_updates:
        issues.append(f"Invalid update rule: {UPDATE_RULE}. Must be one of {valid_updates}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "network_size": NETWORK_SIZE,
        "theoretical_capacity": theoretical_capacity,
        "max_patterns": MAX_PATTERNS
    }

def get_experiment_config(experiment_type: str) -> Dict[str, Any]:
    """Get configuration for specific experiment type."""
    
    base_config = {
        "network_size": NETWORK_SIZE,
        "pattern_size": PATTERN_SIZE,
        "max_iterations": MAX_ITERATIONS,
        "convergence_threshold": CONVERGENCE_THRESHOLD
    }
    
    if experiment_type == "capacity":
        return {
            **base_config,
            "sizes": CAPACITY_EXPERIMENT_SIZES,
            "trials": CAPACITY_TRIALS,
            "patterns": range(1, MAX_PATTERNS + 1)
        }
    
    elif experiment_type == "noise_robustness":
        return {
            **base_config,
            "noise_levels": NOISE_EXPERIMENT_LEVELS,
            "trials": NOISE_TRIALS,
            "patterns": min(5, MAX_PATTERNS)  # Use subset for noise tests
        }
    
    elif experiment_type == "convergence":
        return {
            **base_config,
            "trials": CONVERGENCE_TRIALS,
            "timeout": CONVERGENCE_TIMEOUT,
            "track_energy": True
        }
    
    else:
        return base_config

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def print_config_summary() -> None:
    """Print a summary of the current configuration."""
    
    validation = validate_config()
    
    print("="*80)
    print("HOPFIELD NETWORK CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Network Architecture:")
    print(f"  - Size: {NETWORK_SIZE} neurons ({PATTERN_HEIGHT}x{PATTERN_WIDTH})")
    print(f"  - Update Rule: {UPDATE_RULE}")
    print(f"  - Symmetric Weights: {SYMMETRIC_WEIGHTS}")
    print(f"  - Zero Diagonal: {ZERO_DIAGONAL}")
    print()
    print(f"Memory Capacity:")
    print(f"  - Theoretical Capacity: ~{int(CAPACITY_RATIO * NETWORK_SIZE)} patterns")
    print(f"  - Max Patterns to Store: {MAX_PATTERNS}")
    print(f"  - Capacity Ratio: {CAPACITY_RATIO}")
    print()
    print(f"Energy Dynamics:")
    print(f"  - Max Iterations: {MAX_ITERATIONS}")
    print(f"  - Convergence Threshold: {CONVERGENCE_THRESHOLD}")
    print(f"  - Temperature: {TEMPERATURE}")
    print()
    print(f"Validation Status: {'✓ VALID' if validation['valid'] else '✗ ISSUES'}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    print("="*80)

# Validate configuration on import
if __name__ == "__main__":
    print_config_summary()
