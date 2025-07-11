"""
Shared Visualization Framework for AI-From-Scratch-to-Scale
==========================================================

This module provides reusable visualization components for educational AI/ML implementations.
Designed to maintain consistency across all models while providing educational value.

Key Features:
- Consistent styling and color schemes across all models
- Educational annotations and mathematical context
- Reusable base classes for model-specific visualizers
- Common ML visualization patterns (confusion matrices, training curves, etc.)
- Professional-quality plots suitable for educational materials

Architecture:
- BaseVisualizer: Core functionality all models inherit
- Common components: Reusable visualization functions
- Educational framework: Consistent annotations and learning context
- Styling system: Unified themes and color schemes

Example Usage:
    from ai_from_scratch_shared.visualization import BaseVisualizer, ConfusionMatrixVisualizer
    
    class PerceptronVisualizer(BaseVisualizer):
        def __init__(self):
            super().__init__(model_name="Perceptron")
            self.confusion_matrix = ConfusionMatrixVisualizer()
"""

from .base import BaseVisualizer
from .common import (
    ConfusionMatrixVisualizer,
    TrainingCurveVisualizer,
    DecisionBoundaryVisualizer,
    DataDistributionVisualizer
)
from .style import (
    EDUCATIONAL_COLORS,
    FIGURE_SIZES,
    apply_educational_theme,
    apply_professional_theme,
    get_model_color_scheme
)
from .utils import (
    save_and_show_plot,
    create_figure_with_theme,
    add_educational_annotation,
    format_axes_for_education
)
from .educational import (
    EducationalAnnotator,
    add_mathematical_context,
    add_performance_insights,
    create_concept_explanation
)

__version__ = "1.0.0"
__author__ = "AI-From-Scratch-to-Scale Project"

# Public API exports
__all__ = [
    # Core classes
    "BaseVisualizer",
    "ConfusionMatrixVisualizer", 
    "TrainingCurveVisualizer",
    "DecisionBoundaryVisualizer",
    "DataDistributionVisualizer",
    "EducationalAnnotator",
    
    # Styling and themes
    "EDUCATIONAL_COLORS",
    "FIGURE_SIZES",
    "apply_educational_theme",
    "apply_professional_theme", 
    "get_model_color_scheme",
    
    # Utility functions
    "save_and_show_plot",
    "create_figure_with_theme",
    "add_educational_annotation",
    "format_axes_for_education",
    "add_mathematical_context",
    "add_performance_insights",
    "create_concept_explanation"
]
