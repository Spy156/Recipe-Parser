import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from datasets import load_dataset
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)

# Image and model configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

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

# Function to preprocess the image data
def preprocess_image(example):
    image = example['image'].resize(IMG_SIZE)
    # Convert grayscale to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = np.array(image, dtype=np.float32)
    image = preprocess_input(image)
    label = tf.one_hot(example['label'], depth=num_classes)
    return image, label

# Convert dataset to TensorFlow dataset
def to_tf_dataset(dataset):
    return tf.data.Dataset.from_generator(
        lambda: map(preprocess_image, dataset),
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
        )
    ).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Convert train and validation datasets
train_tf_dataset = to_tf_dataset(train_ds)
validation_tf_dataset = to_tf_dataset(validation_ds)

# Inspect a few samples from the dataset
for images, labels in train_tf_dataset.take(1):
    for i in range(3):
        plt.figure(figsize=(5,5))
        plt.imshow(images[i] / 2 + 0.5)  # De-normalize the image
        plt.title(f"Shape: {images[i].shape}")
        plt.axis('off')
        plt.show()

# Create the model
def create_food_classification_model(input_shape, num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)  # Lower learning rate and gradient clipping
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create and compile the model
with tf.device('/GPU:0'):
    model = create_food_classification_model((*IMG_SIZE, 3), num_classes)

model.summary()

# Callbacks
model_checkpoint = ModelCheckpoint(
    'food_classification_best_model.h5', save_best_only=True, monitor='val_loss', verbose=1
)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

class MonitorCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        logging.info(f"Starting epoch {epoch + 1}")

    def on_epoch_end(self, epoch, logs=None):
        logging.info(f"Epoch {epoch + 1}/{EPOCHS}")
        for k, v in logs.items():
            logging.info(f"{k}: {v:.4f}")

# Training the model
try:
    logging.info("Training the model...")
    with tf.device('/GPU:0'):
        history = model.fit(
            train_tf_dataset,
            validation_data=validation_tf_dataset,
            epochs=EPOCHS,
            callbacks=[early_stopping, reduce_lr, model_checkpoint, MonitorCallback()]
        )

    # Save class names
    class_names = train_ds.features['label'].names
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

except tf.errors.InvalidArgumentError as e:
    logging.error(f"InvalidArgumentError occurred: {str(e)}")
    logging.error("This might be due to inconsistent image shapes in the dataset.")
    # You might want to add additional error handling or data inspection here
except Exception as e:
    logging.error(f"An error occurred during training: {str(e)}")
    raise

# Save the final model
model.save('food_classification_final_model.h5')
logging.info("Final model saved as 'food_classification_final_model.h5'")