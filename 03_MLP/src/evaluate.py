# -*- coding: utf-8 -*-
"""Evaluation and testing utilities for the MLP model."""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)


def evaluate_model(
    model, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    class_names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data and return comprehensive metrics.
    
    Args:
        model: Trained MLP model with predict method
        X_test: Test features of shape (n_samples, n_features)
        y_test: True labels (can be one-hot encoded or class indices)
        class_names: Optional list of class names for labeling
        
    Returns:
        Dictionary containing evaluation metrics
        
    Raises:
        ValueError: If input dimensions are incorrect
    """
    if X_test.ndim != 2:
        raise ValueError(f"X_test must be 2D array, got {X_test.ndim}D")
    
    logger.info(f"Evaluating model on {X_test.shape[0]} test samples")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Convert one-hot encoded labels to class indices if needed
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test.flatten()
    
    # Ensure predictions are 1D
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Determine if binary or multi-class
    n_classes = len(np.unique(y_true))
    is_binary = n_classes == 2
    
    # Calculate precision, recall, F1
    if is_binary:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Prepare results
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'n_samples': int(X_test.shape[0]),
        'n_classes': int(n_classes),
        'is_binary': is_binary
    }
    
    if class_names is not None:
        results['class_names'] = class_names
    
    # Calculate per-class metrics for multi-class
    if not is_binary and n_classes > 2:
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        results['per_class_metrics'] = {
            'precision': per_class_precision.tolist(),
            'recall': per_class_recall.tolist(),
            'f1_score': per_class_f1.tolist()
        }
    
    logger.info(f"Evaluation completed - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return results


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """
    Print a formatted evaluation report.
    
    Args:
        results: Results dictionary from evaluate_model function
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)
    
    print(f"Test Samples: {results['n_samples']}")
    print(f"Number of Classes: {results['n_classes']}")
    print(f"Classification Type: {'Binary' if results['is_binary'] else 'Multi-class'}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1_score']:.4f}")
    
    # Print per-class metrics if available
    if 'per_class_metrics' in results:
        print(f"\nPer-Class Metrics:")
        class_names = results.get('class_names', [f'Class {i}' for i in range(results['n_classes'])])
        
        for i, class_name in enumerate(class_names):
            if i < len(results['per_class_metrics']['precision']):
                print(f"  {class_name}:")
                print(f"    Precision: {results['per_class_metrics']['precision'][i]:.4f}")
                print(f"    Recall:    {results['per_class_metrics']['recall'][i]:.4f}")
                print(f"    F1-Score:  {results['per_class_metrics']['f1_score'][i]:.4f}")
    
    print("\nConfusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    class_names = results.get('class_names', [f'Class {i}' for i in range(results['n_classes'])])
    
    # Print header
    print("    ", end="")
    for name in class_names:
        print(f"{name:>8}", end="")
    print()
    
    # Print matrix with row labels
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>4}", end="")
        for val in row:
            print(f"{val:>8}", end="")
        print()
    
    print("="*50)


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
