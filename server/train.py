import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from datasets import load_dataset
import logging
import numpy as np
import random
import os
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Enable mixed precision for faster training on modern GPUs
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Image and model configuration
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 2

# Check TensorFlow GPU support
logging.info(f"TensorFlow version: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    logging.info(f"Detected {len(gpu_devices)} GPU(s): {gpu_devices}")
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    logging.warning("No GPU detected, training on CPU.")

# Load the Food101 dataset
logging.info("Loading the Food101 dataset...")
try:
    ds = load_dataset("food101", split=['train', 'validation'])
    train_ds, validation_ds = ds
    num_classes = train_ds.features['label'].num_classes

    logging.info(f"Training set size: {len(train_ds)}, Validation set size: {len(validation_ds)}")
except Exception as e:
    logging.error(f"Error loading dataset: {str(e)}")
    raise

# Data augmentation for training
def augment(image):
    return tf.image.random_flip_left_right(
        tf.image.random_brightness(
            tf.image.random_contrast(image, lower=0.8, upper=1.2), max_delta=0.2
        )
    )

# Preprocess images
def preprocess_image(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = augment(image)
    return tf.cast(image, tf.float32) / 255.0, tf.one_hot(label, num_classes)

# Convert dataset to TensorFlow dataset
def to_tf_dataset(dataset, is_train=True):
    ds = tf.data.Dataset.from_tensor_slices(([example['image'] for example in dataset], [example['label'] for example in dataset]))
    if is_train:
        ds = ds.shuffle(10000, reshuffle_each_iteration=True)
    return ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_tf_dataset = to_tf_dataset(train_ds, is_train=True)
validation_tf_dataset = to_tf_dataset(validation_ds, is_train=False)

# Create the model with MobileNetV2
def create_food_classification_model(input_shape, num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Create and compile the model
with tf.device('/GPU:0'):
    model = create_food_classification_model((*IMG_SIZE, 3), num_classes)

# Compile the model
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps=1000, decay_rate=0.9, staircase=True)
model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    ModelCheckpoint('food_classification_best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=2, min_lr=1e-6),
]

class DetailedLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, num_epochs):
        super().__init__()
        self.num_epochs = num_epochs

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        logging.info(f"Starting training for {self.num_epochs} epochs")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        logging.info(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {logs['loss']:.4f}, Acc: {logs['accuracy']:.4f}, Val Loss: {logs['val_loss']:.4f}, Val Acc: {logs['val_accuracy']:.4f} ({elapsed_time:.2f}s)")

callbacks.append(DetailedLoggingCallback(EPOCHS))

# Training the model
logging.info("Training the model...")
try:
    with tf.device('/GPU:0'):
        history = model.fit(train_tf_dataset, validation_data=validation_tf_dataset, epochs=EPOCHS, callbacks=callbacks)

    logging.info("Training completed, saving model...")
    model.save('food_classification_final_model.h5')
    logging.info("Final model saved as 'food_classification_final_model.h5'")

except Exception as e:
    logging.error(f"Training failed: {str(e)}")
    raise
