"""
Comparative Analysis of Classical vs. Quantum Neural Networks for Image Classification

This paper presents a comprehensive comparison between classical and quantum neural network architectures
for image classification tasks, evaluating their performance on the Fashion-MNIST dataset.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime

# Set figure aesthetics for the paper
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

# Create directory for paper figures
os.makedirs('paper_figures', exist_ok=True)

def generate_paper_figures():
    """Generate and save high-quality figures for the academic paper."""
    
    # Load results if they exist
    try:
        comparison_df = pd.read_csv('results/model_comparison.csv')
        
        # Figure 1: Accuracy Comparison
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Model', y='Accuracy', data=comparison_df, palette='viridis')
        ax.set_title('Accuracy Comparison of Classical vs. Quantum Models', fontsize=16)
        ax.set_ylim(0, 1)
        ax.bar_label(ax.containers[0], fmt='%.3f', fontsize=12)
        plt.tight_layout()
        plt.savefig('paper_figures/fig1_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        
        # Figure 2: Training Time Comparison
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Model', y='Training Time (s)', data=comparison_df, palette='viridis')
        ax.set_title('Training Time Comparison', fontsize=16)
        ax.bar_label(ax.containers[0], fmt='%.1f s', fontsize=12)
        plt.tight_layout()
        plt.savefig('paper_figures/fig2_training_time.png', dpi=300, bbox_inches='tight')
        
        # Figure 3: Inference Time Comparison
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Model', y='Inference Time per Sample (ms)', data=comparison_df, palette='viridis')
        ax.set_title('Inference Time per Sample', fontsize=16)
        ax.bar_label(ax.containers[0], fmt='%.2f ms', fontsize=12)
        plt.tight_layout()
        plt.savefig('paper_figures/fig3_inference_time.png', dpi=300, bbox_inches='tight')
        
        # Figure 4: Model Size Comparison
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Model', y='Model Size (params)', data=comparison_df, palette='viridis')
        ax.set_title('Model Size Comparison', fontsize=16)
        ax.bar_label(ax.containers[0], fmt='%d', fontsize=12)
        plt.tight_layout()
        plt.savefig('paper_figures/fig4_model_size.png', dpi=300, bbox_inches='tight')
        
        # Figure 5: Combined Performance Metrics
        # Normalize metrics for radar chart
        metrics = ['Accuracy', 'Training Time (s)', 'Inference Time per Sample (ms)', 'Model Size (params)']
        normalized_df = comparison_df.copy()
        
        # For accuracy, higher is better, so no inversion needed
        normalized_df['Accuracy'] = normalized_df['Accuracy'] / normalized_df['Accuracy'].max()
        
        # For other metrics, lower is better, so invert the normalization
        for metric in metrics[1:]:
            normalized_df[metric] = 1 - (normalized_df[metric] / normalized_df[metric].max())
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Set the angles for each metric
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Plot each model
        for i, model in enumerate(normalized_df['Model']):
            values = normalized_df.loc[i, metrics].values.tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels and title
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title('Normalized Performance Metrics', fontsize=16)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.savefig('paper_figures/fig5_radar_chart.png', dpi=300, bbox_inches='tight')
        
        return True
    except Exception as e:
        print(f"Error generating paper figures: {e}")
        return False

if __name__ == "__main__":
    generate_paper_figures()
