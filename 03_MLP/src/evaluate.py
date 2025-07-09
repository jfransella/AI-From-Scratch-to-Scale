# -*- coding: utf-8 -*-
"""Evaluation and testing utilities for the MLP model.

This module provides comprehensive evaluation capabilities for neural networks,
demonstrating professional ML evaluation practices including:
- Multi-metric evaluation beyond simple accuracy
- Confusion matrix analysis for understanding misclassifications
- Robustness testing for real-world deployment readiness
- Per-class performance analysis for imbalanced datasets

Educational Context:
    Model evaluation is crucial for understanding:
    1. **Accuracy**: Overall correctness (can be misleading for imbalanced data)
    2. **Precision**: Of predicted positives, how many are actually positive?
    3. **Recall**: Of actual positives, how many did we correctly identify?
    4. **F1-Score**: Harmonic mean of precision and recall
    5. **Confusion Matrix**: Detailed breakdown of prediction errors
    6. **Robustness**: How well does the model handle real-world variations?
    
    These metrics help identify model weaknesses and guide improvements.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

# Constants for evaluation
MIN_SAMPLES_FOR_METRICS: int = 1  # Minimum samples needed for meaningful metrics
ROBUSTNESS_SCORE_MAX: float = 1.0  # Maximum possible robustness score


def evaluate_model(
    model, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    class_names: Optional[list] = None
) -> Dict[str, Any]:
    """Evaluate a trained model on test data and return comprehensive metrics.
    
    Educational Context:
        This function demonstrates comprehensive model evaluation beyond accuracy:
        
        **Why Multiple Metrics Matter:**
        - Accuracy can be misleading for imbalanced datasets (90% accuracy with 90% class 0)
        - Precision answers: "When I predict positive, how often am I right?"
        - Recall answers: "How many actual positives did I find?"
        - F1-score balances precision and recall (harmonic mean)
        - Confusion matrix shows exactly where the model makes mistakes
        
        **Binary vs Multi-class:**
        - Binary: Direct calculation of precision/recall/F1
        - Multi-class: Use weighted average to handle class imbalance
    
    Args:
        model: Trained MLP model with predict method
        X_test: Test features of shape (n_samples, n_features)
        y_test: True labels (can be one-hot encoded or class indices)
        class_names: Optional list of class names for labeling results
        
    Returns:
        Dictionary containing comprehensive evaluation metrics:
        - accuracy: Overall fraction of correct predictions
        - precision: Weighted average precision across classes
        - recall: Weighted average recall across classes  
        - f1_score: Weighted average F1-score across classes
        - confusion_matrix: Detailed prediction breakdown
        - per_class_metrics: Individual metrics for each class (multi-class only)
        - n_samples: Number of test samples
        - n_classes: Number of unique classes
        - is_binary: Whether this is binary classification
        
    Raises:
        ValueError: If input dimensions are incorrect or data is invalid
    """
    # Validate input format and dimensions
    if X_test.ndim != 2:
        raise ValueError(f"X_test must be 2D array, got {X_test.ndim}D")
    if X_test.shape[0] < MIN_SAMPLES_FOR_METRICS:
        raise ValueError(f"Need at least {MIN_SAMPLES_FOR_METRICS} samples for evaluation")
    
    logger.info(f"🔍 Evaluating model on {X_test.shape[0]} test samples...")
    
    # Get model predictions
    y_pred = model.predict(X_test)
    
    # Convert one-hot encoded labels to class indices if needed
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
        logger.debug("Converted one-hot encoded labels to class indices")
    else:
        y_true = y_test.flatten()
    
    # Ensure predictions are 1D for metric calculation
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    
    # Determine problem type
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)
    is_binary = n_classes == 2
    
    logger.info(f"📊 Problem type: {'Binary' if is_binary else 'Multi-class'} ({n_classes} classes)")
    
    # Calculate core metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Choose appropriate averaging strategy based on problem type
    if is_binary:
        # Binary classification: use standard metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        # Multi-class: use weighted average to handle class imbalance
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate confusion matrix for detailed error analysis
    cm = confusion_matrix(y_true, y_pred)
    
    # Prepare comprehensive results dictionary
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'n_samples': int(X_test.shape[0]),
        'n_classes': int(n_classes),
        'is_binary': is_binary,
        'unique_classes': unique_classes.tolist()
    }
    
    # Add class names if provided
    if class_names is not None:
        if len(class_names) >= n_classes:
            results['class_names'] = class_names[:n_classes]
        else:
            logger.warning(f"⚠️  Provided {len(class_names)} class names but found {n_classes} classes")
            results['class_names'] = class_names + [f'Class_{i}' for i in range(len(class_names), n_classes)]
    
    # Calculate per-class metrics for multi-class problems
    if not is_binary and n_classes > 2:
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        results['per_class_metrics'] = {
            'precision': per_class_precision.tolist(),
            'recall': per_class_recall.tolist(),
            'f1_score': per_class_f1.tolist()
        }
        
        logger.debug(f"Calculated per-class metrics for {n_classes} classes")
    
    logger.info(f"✅ Evaluation completed - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return results


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """Print a comprehensive, formatted evaluation report.
    
    Educational Context:
        A well-formatted evaluation report helps with:
        1. **Quick Assessment**: Key metrics at a glance
        2. **Error Analysis**: Confusion matrix shows where mistakes occur
        3. **Class-wise Performance**: Identify which classes need improvement
        4. **Model Debugging**: Understanding failure modes
        
        This report format is commonly used in ML research and industry.
    
    Args:
        results: Results dictionary from evaluate_model function
    """
    print("\n" + "="*60)
    print("🎯 MODEL EVALUATION REPORT")
    print("="*60)
    
    # Basic dataset information
    print(f"📊 Dataset Information:")
    print(f"   Test Samples: {results['n_samples']:,}")
    print(f"   Number of Classes: {results['n_classes']}")
    print(f"   Classification Type: {'Binary' if results['is_binary'] else 'Multi-class'}")
    
    # Overall performance metrics
    print(f"\n📈 Overall Performance:")
    print(f"   Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1-Score:  {results['f1_score']:.4f}")
    
    # Performance interpretation
    if results['accuracy'] >= 0.95:
        performance_level = "Excellent ⭐⭐⭐"
    elif results['accuracy'] >= 0.90:
        performance_level = "Very Good ⭐⭐"
    elif results['accuracy'] >= 0.80:
        performance_level = "Good ⭐"
    elif results['accuracy'] >= 0.70:
        performance_level = "Fair"
    else:
        performance_level = "Needs Improvement"
    
    print(f"   Performance Level: {performance_level}")
    
    # Per-class metrics for multi-class problems
    if 'per_class_metrics' in results:
        print(f"\n📋 Per-Class Performance:")
        class_names = results.get('class_names', [f'Class_{i}' for i in range(results['n_classes'])])
        
        # Create formatted table header
        print(f"   {'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print(f"   {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
        
        for i, class_name in enumerate(class_names):
            if i < len(results['per_class_metrics']['precision']):
                precision = results['per_class_metrics']['precision'][i]
                recall = results['per_class_metrics']['recall'][i]
                f1_score = results['per_class_metrics']['f1_score'][i]
                
                # Truncate long class names for formatting
                display_name = class_name[:14] if len(class_name) > 14 else class_name
                
                print(f"   {display_name:<15} {precision:<10.4f} {recall:<10.4f} {f1_score:<10.4f}")
    
    # Confusion Matrix Analysis
    print(f"\n🔍 Confusion Matrix Analysis:")
    cm = np.array(results['confusion_matrix'])
    class_names = results.get('class_names', [f'Class_{i}' for i in range(results['n_classes'])])
    
    # Print header with class names
    print("      Predicted →")
    print("   ", end="")
    for name in class_names:
        print(f"{name[:8]:>9}", end="")
    print()
    
    # Print matrix with row labels
    print("   Actual ↓")
    for i, row in enumerate(cm):
        class_name = class_names[i][:8] if i < len(class_names) else f"Class_{i}"
        print(f"   {class_name:>8}", end="")
        for val in row:
            print(f"{val:>9}", end="")
        print()
    
    # Confusion Matrix Insights
    if results['n_classes'] == 2:
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        print(f"\n   Binary Classification Breakdown:")
        print(f"   True Positives:  {tp:>4}")
        print(f"   False Positives: {fp:>4}")
        print(f"   True Negatives:  {tn:>4}")
        print(f"   False Negatives: {fn:>4}")
        
        if tp + fn > 0:
            print(f"   Sensitivity (Recall): {tp/(tp+fn):.4f}")
        if tn + fp > 0:
            print(f"   Specificity:          {tn/(tn+fp):.4f}")
    
    # Model insights and recommendations
    print(f"\n💡 Model Insights:")
    
    # Check for class imbalance issues
    if 'per_class_metrics' in results:
        f1_scores = results['per_class_metrics']['f1_score']
        min_f1 = min(f1_scores)
        max_f1 = max(f1_scores)
        f1_variance = max_f1 - min_f1
        
        if f1_variance > 0.2:
            print(f"   ⚠️  High F1-score variance ({f1_variance:.3f}) indicates class imbalance")
            worst_class_idx = f1_scores.index(min_f1)
            worst_class = class_names[worst_class_idx] if worst_class_idx < len(class_names) else f"Class_{worst_class_idx}"
            print(f"   📉 Worst performing class: {worst_class} (F1: {min_f1:.3f})")
        else:
            print(f"   ✅ Balanced performance across classes (F1 variance: {f1_variance:.3f})")
    
    # Overall assessment
    if results['accuracy'] > 0.95 and results['f1_score'] > 0.95:
        print(f"   🎉 Excellent model performance! Ready for deployment.")
    elif results['accuracy'] > 0.85:
        print(f"   👍 Good model performance. Consider fine-tuning for production.")
    else:
        print(f"   🔧 Model needs improvement. Consider:")
        print(f"      - More training data")
        print(f"      - Feature engineering")
        print(f"      - Architecture adjustments")
        print(f"      - Hyperparameter tuning")
    
    print("="*60 + "\n")


def calculate_model_robustness(
    model,
    X_original: np.ndarray,
    X_modified: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Calculate robustness metrics by comparing performance on original vs modified data.
    
    Args:
        model: Trained model with predict method
        X_original: Original test data
        X_modified: Modified test data (e.g., with noise, shifts, etc.)
        y_test: True labels
        
    Returns:
        Dictionary containing robustness metrics
    """
    logger.info("Calculating model robustness metrics")
    
    # Evaluate on original data
    original_results = evaluate_model(model, X_original, y_test)
    
    # Evaluate on modified data
    modified_results = evaluate_model(model, X_modified, y_test)
    
    # Calculate robustness metrics
    accuracy_drop = original_results['accuracy'] - modified_results['accuracy']
    relative_accuracy_drop = accuracy_drop / original_results['accuracy'] if original_results['accuracy'] > 0 else 0
    
    robustness_metrics = {
        'original_accuracy': original_results['accuracy'],
        'modified_accuracy': modified_results['accuracy'],
        'accuracy_drop': float(accuracy_drop),
        'relative_accuracy_drop': float(relative_accuracy_drop),
        'robustness_score': float(1.0 - abs(relative_accuracy_drop))  # Higher is better
    }
    
    logger.info(f"Robustness analysis - Accuracy drop: {accuracy_drop:.4f} "
                f"({relative_accuracy_drop*100:.1f}%)")
    
    return robustness_metrics
