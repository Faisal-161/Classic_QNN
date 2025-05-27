"""
Implementation of quantum neural network models for image classification.
"""

import pennylane as qml
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import time
from typing import Dict, Tuple, Any, Optional

def create_quantum_circuit(n_qubits, n_layers, observables=None):
    """
    Create a variational quantum circuit for image classification.
    
    Args:
        n_qubits: Number of qubits in the circuit
        n_layers: Number of variational layers
        observables: List of observables to measure (default: Pauli-Z on each qubit)
        
    Returns:
        Quantum circuit function
    """
    # Set default observables if not provided
    if observables is None:
        observables = [qml.PauliZ(i) for i in range(n_qubits)]
    
    # Create a quantum device
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="tf")
    def quantum_circuit(inputs, weights):
        """
        Variational quantum circuit with data re-uploading structure.
        
        Args:
            inputs: Input data to encode
            weights: Trainable weights for the circuit
            
        Returns:
            Expectation values of the observables
        """
        # Normalize inputs
        inputs = tf.clip_by_value(inputs, -np.pi, np.pi)
        
        # Reshape weights for easier handling
        weights = tf.reshape(weights, (n_layers, n_qubits, 3))
        
        # Initial data encoding
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # Variational layers
        for layer in range(n_layers):
            # Rotation gates with trainable parameters
            for qubit in range(n_qubits):
                qml.RX(weights[layer, qubit, 0], wires=qubit)
                qml.RY(weights[layer, qubit, 1], wires=qubit)
                qml.RZ(weights[layer, qubit, 2], wires=qubit)
            
            # Entanglement
            for qubit in range(n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
            
            # Connect last qubit to first for circular entanglement
            if n_qubits > 1:
                qml.CNOT(wires=[n_qubits - 1, 0])
            
            # Re-upload data if not the last layer
            if layer < n_layers - 1:
                for i in range(n_qubits):
                    qml.RY(inputs[i], wires=i)
        
        # Return expectation values
        return [qml.expval(obs) for obs in observables]
    
    return quantum_circuit

class QuantumLayer(tf.keras.layers.Layer):
    """
    Keras layer implementing a variational quantum circuit.
    """
    def __init__(self, n_qubits, n_layers, observables=None, **kwargs):
        super(QuantumLayer, self).__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.observables = observables
        self.quantum_circuit = create_quantum_circuit(n_qubits, n_layers, observables)
        
        # Initialize weights
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.weight_shapes = weight_shapes
        
        # Initialize trainable variables
        self.weights_var = tf.Variable(
            tf.random.uniform(shape=tf.TensorShape(weight_shapes["weights"]), 
                             minval=0, maxval=2*np.pi),
            trainable=True, name="quantum_weights"
        )
    
    def call(self, inputs):
        """
        Forward pass through the quantum layer.
        
        Args:
            inputs: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Output tensor of shape (batch_size, n_qubits)
        """
        batch_size = tf.shape(inputs)[0]
        
        # Process each input in the batch
        output = tf.TensorArray(dtype=tf.float32, size=batch_size)
        
        for i in range(batch_size):
            # Select the i-th input
            input_i = inputs[i]
            
            # Ensure input has the right shape for the quantum circuit
            input_padded = tf.pad(input_i, [[0, max(0, self.n_qubits - tf.shape(input_i)[0])]])
            input_truncated = input_padded[:self.n_qubits]
            
            # Apply quantum circuit
            output_i = self.quantum_circuit(input_truncated, self.weights_var)
            output = output.write(i, output_i)
        
        return output.stack()
    
    def get_config(self):
        config = super(QuantumLayer, self).get_config()
        config.update({
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "observables": self.observables
        })
        return config

def create_hybrid_quantum_model(input_shape=(28, 28, 1), n_qubits=4, n_layers=2, num_classes=10):
    """
    Create a hybrid quantum-classical model for image classification.
    
    Args:
        input_shape: Shape of input images
        n_qubits: Number of qubits in the quantum circuit
        n_layers: Number of variational layers
        num_classes: Number of output classes
        
    Returns:
        Compiled hybrid quantum-classical model
    """
    # Classical preprocessing layers
    model = models.Sequential()
    
    # Convolutional feature extraction
    model.add(layers.Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten and reduce dimensionality
    model.add(layers.Flatten())
    model.add(layers.Dense(n_qubits, activation='tanh'))  # Map to [-1, 1] range
    
    # Scale to appropriate range for quantum circuit
    model.add(layers.Lambda(lambda x: x * np.pi))  # Scale to [-π, π]
    
    # Quantum layer
    model.add(QuantumLayer(n_qubits=n_qubits, n_layers=n_layers))
    
    # Output classification layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_pure_quantum_model(input_shape=(28, 28, 1), n_qubits=8, n_layers=3, num_classes=10):
    """
    Create a more quantum-focused model with minimal classical preprocessing.
    
    Args:
        input_shape: Shape of input images
        n_qubits: Number of qubits in the quantum circuit
        n_layers: Number of variational layers
        num_classes: Number of output classes
        
    Returns:
        Compiled quantum-focused model
    """
    # Classical preprocessing layers
    model = models.Sequential()
    
    # Minimal preprocessing - just flatten and reduce dimensionality
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(n_qubits, activation='tanh'))  # Map to [-1, 1] range
    
    # Scale to appropriate range for quantum circuit
    model.add(layers.Lambda(lambda x: x * np.pi))  # Scale to [-π, π]
    
    # Quantum layer
    model.add(QuantumLayer(n_qubits=n_qubits, n_layers=n_layers))
    
    # Output classification layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_quantum_model(
    model, 
    x_train, 
    y_train, 
    x_val=None, 
    y_val=None, 
    batch_size=16, 
    epochs=15,
    patience=3
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Train a quantum neural network model.
    
    Args:
        model: Model to train
        x_train: Training data
        y_train: Training labels
        x_val: Validation data
        y_val: Validation labels
        batch_size: Batch size for training (smaller for quantum models)
        epochs: Maximum number of epochs
        patience: Early stopping patience
        
    Returns:
        Tuple of (trained model, training history, training time)
    """
    callbacks = []
    
    # Add early stopping if validation data is provided
    if x_val is not None and y_val is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        )
    
    # Measure training time
    start_time = time.time()
    
    # Train the model
    if x_val is not None and y_val is not None:
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    return model, history, training_time
