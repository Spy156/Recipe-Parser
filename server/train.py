import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from datasets import load_dataset
import logging
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)

# Image and model configuration
IMG_SIZE = (224, 224)  # You can also reduce this to (160, 160) or (128, 128)
BATCH_SIZE = 32  # Adjust based on available memory
EPOCHS = 10  # Reduced from 50
AUTOTUNE = tf.data.AUTOTUNE

# Check TensorFlow GPU support
logging.info(f"TensorFlow version: {tf.__version__}")
logging.info(f"Is built with CUDA: {tf.test.is_built_with_cuda()}")
logging.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    logging.info(f"TensorFlow detected {len(physical_devices)} GPU(s): {physical_devices}")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    logging.warning("No GPU detected, training will use the CPU.")

# Load the Food101 dataset
logging.info("Loading the Food101 dataset...")
try:
    ds = load_dataset("food101", split=['train', 'validation'])
    train_ds = ds[0]
    validation_ds = ds[1]
    num_classes = train_ds.features['label'].num_classes

    logging.info(f"Training set size: {len(train_ds)}")
    logging.info(f"Validation set size: {len(validation_ds)}")
except Exception as e:
    logging.error(f"Error loading dataset: {str(e)}")
    raise

# Data augmentation
def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image

# Function to preprocess the image data
def preprocess_image(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    image = augment(image)
    label = tf.one_hot(label, depth=num_classes)
    return image, label

# Convert dataset to TensorFlow dataset
def to_tf_dataset(dataset, is_train=True):
    tf_dataset = tf.data.Dataset.from_tensor_slices((
        [example['image'] for example in dataset],
        [example['label'] for example in dataset]
    ))

    if is_train:
        tf_dataset = tf_dataset.shuffle(10000, reshuffle_each_iteration=True)
    
    tf_dataset = tf_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    tf_dataset = tf_dataset.batch(BATCH_SIZE)
    tf_dataset = tf_dataset.prefetch(AUTOTUNE)
    
    return tf_dataset

# Convert train and validation datasets
train_tf_dataset = to_tf_dataset(train_ds, is_train=True).apply(tf.data.experimental.ignore_errors())
validation_tf_dataset = to_tf_dataset(validation_ds, is_train=False).apply(tf.data.experimental.ignore_errors())

# Create the model
def create_food_classification_model(input_shape, num_classes):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model initially
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),  # Reduce Dense layer size for efficiency
        tf.keras.layers.Dropout(0.3),  # Slightly lower dropout for efficiency
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Create and compile the model
with tf.device('/GPU:0'):
    model = create_food_classification_model((*IMG_SIZE, 3), num_classes)

model.summary()

# Compile the model
initial_learning_rate = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True
)

optimizer = Adam(learning_rate=lr_schedule)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
model_checkpoint = ModelCheckpoint(
    'food_classification_best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=3, min_lr=1e-6)

class DetailedLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, num_epochs):
        super().__init__()
        self.num_epochs = num_epochs
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        logging.info(f"Starting training for {self.num_epochs} epochs")

    def on_epoch_begin(self, epoch, logs=None):
        logging.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        logging.info(f"Epoch {epoch + 1}/{self.num_epochs} completed in {epoch_time:.2f} seconds")
        logging.info(f"Epoch {epoch + 1} - Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, "
                     f"Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}")

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        logging.info(f"Training completed in {total_time:.2f} seconds")

# Training the model
try:
    logging.info("Training the model...")
    with tf.device('/GPU:0'):
        history = model.fit(
            train_tf_dataset,
            validation_data=validation_tf_dataset,
            epochs=EPOCHS,
            callbacks=[early_stopping, reduce_lr, model_checkpoint, DetailedLoggingCallback(EPOCHS)]
        )

    logging.info("Model training completed successfully")

    # Save the final model
    model.save('food_classification_final_model.keras')
    logging.info("Final model saved as 'food_classification_final_model.keras'")

except Exception as e:
    logging.error(f"An error occurred during training: {str(e)}")
    raise

logging.info("Training completed!")
