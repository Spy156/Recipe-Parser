import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from custom_model import create_food_classification_model
from datasets import load_dataset
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
tf.random.set_seed(42)

# Image and model configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
N_TRAIN_SAMPLES = 1000  # Number of training samples to take
N_VALIDATION_SAMPLES = 500  # Number of validation samples to take

# Check if GPU is available and log the info
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    logging.info(f"TensorFlow detected {len(physical_devices)} GPU(s): {physical_devices}")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    logging.warning("No GPU detected, training will use the CPU.")

# Load the Food101 dataset without caching to disk
logging.info("Loading the Food101 dataset without caching...")
ds = load_dataset("ethz/food101", split=['train', 'validation'], cache_dir='/tmp/food101')  # cache to a tmp directory

num_classes = ds['train'].features['label'].num_classes

# Reduce the number of samples using random sampling
logging.info(f"Reducing the dataset to {N_TRAIN_SAMPLES} training samples and {N_VALIDATION_SAMPLES} validation samples...")

# Randomly select train and validation sample indices
train_indices = random.sample(range(len(ds['train'])), N_TRAIN_SAMPLES)
validation_indices = random.sample(range(len(ds['validation'])), N_VALIDATION_SAMPLES)

# Select the subset of the dataset
subset_train = ds['train'].select(train_indices)
subset_validation = ds['validation'].select(validation_indices)

# Function to preprocess the image data
def preprocess_image(batch):
    images = batch['image']
    labels = batch['label']
    resized_images = []

    if len(images) != len(labels):
        raise ValueError(f"Mismatch between number of images ({len(images)}) and labels ({len(labels)})")

    for img in images:
        if isinstance(img, Image.Image):
            img = img.resize(IMG_SIZE)
            img = np.array(img, dtype=np.float32)
            img = preprocess_input(img)
            resized_images.append(img)
        else:
            resized_images.append(np.zeros((*IMG_SIZE, 3), dtype=np.float32))

    resized_images = [img for img in resized_images if img.shape == (*IMG_SIZE, 3)]

    if len(resized_images) != len(labels):
        min_length = min(len(resized_images), len(labels))
        resized_images = resized_images[:min_length]
        labels = labels[:min_length]

    if len(resized_images) == 0:
        raise ValueError("No valid images found in the batch after resizing.")

    batch['image'] = np.stack(resized_images)
    batch['label'] = np.array(labels)
    return batch

# Preprocess the subset of the dataset
logging.info("Preprocessing the dataset in batches...")
train_ds = subset_train.map(preprocess_image, batched=True, batch_size=BATCH_SIZE)
validation_ds = subset_validation.map(preprocess_image, batched=True, batch_size=BATCH_SIZE)

# Convert dataset to TensorFlow dataset without storing any intermediate files
def to_tf_dataset(dataset):
    def generator():
        for example in dataset:
            label = tf.one_hot(example['label'], depth=num_classes)
            yield example['image'], label

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
        )
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Convert train and validation datasets
train_tf_dataset = to_tf_dataset(train_ds)
validation_tf_dataset = to_tf_dataset(validation_ds)

# Create the model
logging.info("Creating and compiling the model...")
model = create_food_classification_model((*IMG_SIZE, 3), num_classes)

# Callbacks: Save only the best model and reduce unnecessary checkpoints
model_checkpoint = ModelCheckpoint(
    'food_classification_best_model.h5', save_best_only=True, monitor='val_loss', verbose=1
)

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Training the model with minimal disk usage
try:
    logging.info("Training the model...")
    history = model.fit(
        train_tf_dataset,
        validation_data=validation_tf_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr, model_checkpoint]
    )

    # Save class names to memory-efficient file
    class_names = ds[0].features['label'].names
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    logging.info("Class names saved to 'class_names.txt'")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    logging.info("Training history plot saved as 'training_history.png'")

    # Evaluate the model
    evaluation = model.evaluate(validation_tf_dataset)
    logging.info(f"Validation Loss: {evaluation[0]:.4f}")
    logging.info(f"Validation Accuracy: {evaluation[1]:.4f}")

except Exception as e:
    logging.error(f"An error occurred during training: {str(e)}")
