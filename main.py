"""
Main script for comparative analysis of classical vs. quantum neural networks.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.datasets import mnist, fashion_mnist

# Import our modules
from utils import preprocess_data, plot_training_history, evaluate_model_performance, compare_models, plot_comparison
from classical_models import create_lenet5, create_simple_resnet, train_classical_model
from quantum_models import create_hybrid_quantum_model, create_pure_quantum_model, train_quantum_model

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create results directory
os.makedirs('results', exist_ok=True)

def main():
    """
    Main function to run the comparative analysis.
    """
    print("Loading dataset...")
    # Load Fashion MNIST dataset (more challenging than MNIST)
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # For quantum models, we'll use a smaller subset due to computational constraints
    quantum_subset_size = 1000  # Adjust based on computational resources
    
    # Preprocess data for classical models
    x_train_classical, y_train_classical, x_test_classical, y_test_classical = preprocess_data(
        x_train, y_train, x_test, y_test
    )
    
    # Preprocess data for quantum models (smaller subset)
    x_train_quantum, y_train_quantum, x_test_quantum, y_test_quantum = preprocess_data(
        x_train, y_train, x_test, y_test, subset_size=quantum_subset_size
    )
    
    print(f"Classical dataset shapes: {x_train_classical.shape}, {x_test_classical.shape}")
    print(f"Quantum dataset shapes: {x_train_quantum.shape}, {x_test_quantum.shape}")
    
    # Store results
    results = {}
    
    # Train and evaluate LeNet-5
    print("\nTraining LeNet-5 model...")
    lenet_model = create_lenet5()
    lenet_model, lenet_history, lenet_training_time = train_classical_model(
        lenet_model, x_train_classical, y_train_classical, 
        epochs=10
    )
    
    # Evaluate LeNet-5
    lenet_metrics = evaluate_model_performance(lenet_model, x_test_classical, y_test_classical)
    lenet_metrics['training_time'] = lenet_training_time
    results['LeNet-5'] = lenet_metrics
    
    # Plot and save training history
    plot_training_history(lenet_history, title='LeNet-5')
    plt.savefig('results/lenet5_training_history.png')
    
    # Train and evaluate Simple ResNet
    print("\nTraining Simple ResNet model...")
    resnet_model = create_simple_resnet()
    resnet_model, resnet_history, resnet_training_time = train_classical_model(
        resnet_model, x_train_classical, y_train_classical, 
        epochs=10
    )
    
    # Evaluate Simple ResNet
    resnet_metrics = evaluate_model_performance(resnet_model, x_test_classical, y_test_classical)
    resnet_metrics['training_time'] = resnet_training_time
    results['Simple ResNet'] = resnet_metrics
    
    # Plot and save training history
    plot_training_history(resnet_history, title='Simple ResNet')
    plt.savefig('results/resnet_training_history.png')
    
    # Train and evaluate Hybrid Quantum model
    print("\nTraining Hybrid Quantum model...")
    hybrid_model = create_hybrid_quantum_model(n_qubits=4, n_layers=2)
    hybrid_model, hybrid_history, hybrid_training_time = train_quantum_model(
        hybrid_model, x_train_quantum, y_train_quantum, 
        batch_size=16, epochs=5
    )
    
    # Evaluate Hybrid Quantum model
    hybrid_metrics = evaluate_model_performance(hybrid_model, x_test_quantum, y_test_quantum, batch_size=16)
    hybrid_metrics['training_time'] = hybrid_training_time
    results['Hybrid Quantum'] = hybrid_metrics
    
    # Plot and save training history
    plot_training_history(hybrid_history, title='Hybrid Quantum')
    plt.savefig('results/hybrid_quantum_training_history.png')
    
    # Train and evaluate Pure Quantum model
    print("\nTraining Pure Quantum model...")
    pure_quantum_model = create_pure_quantum_model(n_qubits=6, n_layers=2)
    pure_quantum_model, pure_quantum_history, pure_quantum_training_time = train_quantum_model(
        pure_quantum_model, x_train_quantum, y_train_quantum, 
        batch_size=16, epochs=5
    )
    
    # Evaluate Pure Quantum model
    pure_quantum_metrics = evaluate_model_performance(pure_quantum_model, x_test_quantum, y_test_quantum, batch_size=16)
    pure_quantum_metrics['training_time'] = pure_quantum_training_time
    results['Pure Quantum'] = pure_quantum_metrics
    
    # Plot and save training history
    plot_training_history(pure_quantum_history, title='Pure Quantum')
    plt.savefig('results/pure_quantum_training_history.png')
    
    # Compare all models
    comparison = compare_models(results)
    print("\nModel Comparison:")
    print(comparison)
    
    # Save comparison to CSV
    comparison.to_csv('results/model_comparison.csv', index=False)
    
    # Plot and save comparison
    plot_comparison(comparison, save_path='results/model_comparison.png')
    
    print("\nAnalysis complete. Results saved to 'results' directory.")
    
    return results, comparison

if __name__ == "__main__":
    main()
