# Quantum vs. Classical Neural Networks for Image Classification

This repository contains the implementation of both classical and quantum neural networks for image classification tasks, along with a comparative analysis of their performance.

## Overview

This project implements and compares classical convolutional neural networks (CNNs) and quantum neural networks for image classification on the Fashion-MNIST dataset. The goal is to evaluate the current state of quantum machine learning for image classification and identify potential advantages and limitations compared to classical approaches.

## Repository Structure

- `classical_models.py`: Implementation of classical neural network architectures (LeNet-5 and Simple ResNet)
- `quantum_models.py`: Implementation of quantum neural network architectures (Hybrid and Pure Quantum models)
- `utils.py`: Utility functions for data preprocessing, evaluation, and visualization
- `main.py`: Main script to run the experiments and generate results
- `paper_figures.py`: Script to generate high-quality figures for the academic paper
- `paper.md`: Academic paper describing the methodology, results, and analysis
- `results/`: Directory containing experimental results and figures

## Requirements

- Python 3.8+
- TensorFlow 2.x
- PyTorch
- PennyLane
- Qiskit
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Usage

1. Install the required packages:
   ```
   pip install tensorflow torch torchvision pennylane qiskit matplotlib pandas seaborn
   ```

2. Run the main script to train and evaluate all models:
   ```
   python main.py
   ```

3. Generate paper figures from the results:
   ```
   python paper_figures.py
   ```

4. View the academic paper in Markdown format:
   ```
   cat paper.md
   ```

## Models Implemented

### Classical Models
1. **LeNet-5**: A classic CNN architecture with two convolutional layers followed by fully connected layers
2. **Simple ResNet**: A simplified ResNet-like architecture with residual connections

### Quantum Models
1. **Hybrid Quantum-Classical Model**: Combines classical convolutional preprocessing with a variational quantum circuit
2. **Pure Quantum Model**: Minimizes classical preprocessing, focusing on the quantum circuit for feature extraction

## Results

The repository includes a comprehensive comparison of the models based on:
- Classification accuracy
- Training time
- Inference speed
- Model size (number of parameters)

The results demonstrate the current state of quantum machine learning for image classification and identify potential crossover points where quantum approaches may offer advantages.

## Citation

If you use this code or the findings in your research, please cite:

```
@article{quantum_vs_classical_nn,
  title={Comparative Analysis of Classical vs. Quantum Neural Networks for Image Classification},
  author={Author},
  journal={Quantum Machine Intelligence},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
