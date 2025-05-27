"""
Utility functions for the comparative analysis of classical vs. quantum neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional

def preprocess_data(x_train, y_train, x_test, y_test, num_classes=10, subset_size=None):
    """
    Preprocess the data for neural network training.
    
    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        num_classes: Number of classes
        subset_size: Optional size to reduce dataset for faster quantum computation
        
    Returns:
        Preprocessed data
    """
    # Normalize pixel values to range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN input
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    
    # If subset size is specified, reduce dataset size
    if subset_size is not None:
        x_train = x_train[:subset_size]
        y_train = y_train[:subset_size]
        x_test = x_test[:subset_size//5]  # Use proportionally smaller test set
        y_test = y_test[:subset_size//5]
    
    return x_train, y_train, x_test, y_test

def plot_training_history(history, title='Training History'):
    """
    Plot the training history of a model.
    
    Args:
        history: Training history object
        title: Plot title
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    return plt

def evaluate_model_performance(model, x_test, y_test, batch_size=32):
    """
    Evaluate model performance with timing.
    
    Args:
        model: Trained model
        x_test: Test data
        y_test: Test labels
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with performance metrics
    """
    # Measure inference time
    start_time = time.time()
    predictions = model.predict(x_test, batch_size=batch_size)
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Calculate accuracy
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Multi-class case
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    else:
        # Binary case
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        true_classes = y_test
    
    accuracy = np.mean(predicted_classes == true_classes)
    
    # Calculate model size (approximate)
    model_size = model.count_params()
    
    return {
        'accuracy': accuracy,
        'inference_time': inference_time,
        'inference_time_per_sample': inference_time / len(x_test),
        'model_size': model_size
    }

def compare_models(results: Dict[str, Dict[str, Any]]):
    """
    Compare performance metrics between different models.
    
    Args:
        results: Dictionary with model results
        
    Returns:
        Comparison DataFrame
    """
    comparison = pd.DataFrame({
        'Model': [],
        'Accuracy': [],
        'Training Time (s)': [],
        'Inference Time (s)': [],
        'Inference Time per Sample (ms)': [],
        'Model Size (params)': []
    })
    
    for model_name, metrics in results.items():
        new_row = pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [metrics['accuracy']],
            'Training Time (s)': [metrics['training_time']],
            'Inference Time (s)': [metrics['inference_time']],
            'Inference Time per Sample (ms)': [metrics['inference_time_per_sample'] * 1000],  # Convert to ms
            'Model Size (params)': [metrics['model_size']]
        })
        comparison = pd.concat([comparison, new_row], ignore_index=True)
    
    return comparison

def plot_comparison(comparison: pd.DataFrame, save_path=None):
    """
    Create visualizations comparing model performance.
    
    Args:
        comparison: DataFrame with model comparisons
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    sns.barplot(x='Model', y='Accuracy', data=comparison, ax=axes[0, 0])
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_ylim(0, 1)
    
    # Training time comparison
    sns.barplot(x='Model', y='Training Time (s)', data=comparison, ax=axes[0, 1])
    axes[0, 1].set_title('Training Time (s)')
    
    # Inference time comparison
    sns.barplot(x='Model', y='Inference Time per Sample (ms)', data=comparison, ax=axes[1, 0])
    axes[1, 0].set_title('Inference Time per Sample (ms)')
    
    # Model size comparison
    sns.barplot(x='Model', y='Model Size (params)', data=comparison, ax=axes[1, 1])
    axes[1, 1].set_title('Model Size (parameters)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt
