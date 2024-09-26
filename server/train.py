import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import os
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set random seed for reproducibility
tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)

# Image and model configuration
IMG_SIZE = (160, 160)  # Reduced image size for faster processing and less memory usage
BATCH_SIZE = 32
EPOCHS = 10

# Check TensorFlow GPU support
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected, training will use the CPU.")

# Load the Food101 dataset (use TensorFlow Datasets for efficient loading)
(train_ds, val_ds), info = tf.keras.utils.get_file(
    origin='https://github.com/datasets/food101/archive/master.zip',
    extract=True
)

num_classes = 101  # Number of classes in Food101 dataset

# Data augmentation for training
def preprocess_image(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    label = tf.one_hot(label, depth=num_classes)
    return image, label

# Convert dataset to TensorFlow dataset
def prepare_dataset(dataset, is_train=True):
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        dataset = dataset.shuffle(10000)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

# Preprocess datasets
train_tf_dataset = prepare_dataset(train_ds, is_train=True)
validation_tf_dataset = prepare_dataset(val_ds, is_train=False)

# Create the model with MobileNetV2 (lightweight)
def create_simple_model(input_shape, num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model

    # Add final layers for classification
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Simple classification head
    ])

    return model

# Create and compile the model
model = create_simple_model((*IMG_SIZE, 3), num_classes)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpoint callback to save best model
checkpoint_cb = ModelCheckpoint('simple_food_model.h5', save_best_only=True)

# Train the model with a simpler strategy
model.fit(
    train_tf_dataset,
    validation_data=validation_tf_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb]
)

# Save final model for Flask server usage
model.save('food_classification_flask_model.h5')
