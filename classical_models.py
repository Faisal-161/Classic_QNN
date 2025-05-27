"""
Implementation of classical neural network models for image classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import time
from typing import Dict, Tuple, Any, Optional

def create_lenet5(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a LeNet-5 CNN architecture.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
        
    Returns:
        Compiled LeNet-5 model
    """
    model = models.Sequential([
        # First convolutional layer
        layers.Conv2D(6, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional layer
        layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and fully connected layers
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_simple_resnet(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a simple ResNet-like architecture.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
        
    Returns:
        Compiled simple ResNet model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(32, kernel_size=(3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # First residual block
    residual = x
    x = layers.Conv2D(32, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, residual])
    x = layers.Activation('relu')(x)
    
    # Pooling
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second residual block
    residual = layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1))(x)
    residual = layers.BatchNormalization()(residual)
    
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, residual])
    x = layers.Activation('relu')(x)
    
    # Pooling
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Output layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_classical_model(
    model, 
    x_train, 
    y_train, 
    x_val=None, 
    y_val=None, 
    batch_size=32, 
    epochs=20,
    patience=5
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Train a classical neural network model.
    
    Args:
        model: Model to train
        x_train: Training data
        y_train: Training labels
        x_val: Validation data
        y_val: Validation labels
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        patience: Early stopping patience
        
    Returns:
        Tuple of (trained model, training history, training time)
    """
    callbacks = []
    
    # Add early stopping if validation data is provided
    if x_val is not None and y_val is not None:
        callbacks.append(
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
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
