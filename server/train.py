import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Dense, Dropout, Flatten, AveragePooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.optimizers import SGD
from keras.regularizers import l2
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration for training
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
TRAIN_SIZE_PER_CLASS = 750
TEST_SIZE_PER_CLASS = 250

# Learning rate scheduler
def schedule(epoch):
    if epoch < 5:
        return 0.001
    elif epoch < 10:
        return 0.0002
    elif epoch < 15:
        return 0.00002
    else:
        return 0.0000005

# Load the Food101 dataset and split it into train/test
def load_and_split_dataset():
    dataset = load_dataset("food101")
    train_dataset = dataset['train']
    test_dataset = dataset['validation']

    # Take 750 samples per class for training and 250 for testing
    train_samples = train_dataset.shuffle(seed=42).select(range(TRAIN_SIZE_PER_CLASS))
    test_samples = test_dataset.shuffle(seed=42).select(range(TEST_SIZE_PER_CLASS))

    return train_samples, test_samples

# Preprocessing function
def preprocess_image(example):
    image = example['image']
    label = example['label']

    # Resize and normalize images
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    label = tf.one_hot(label, depth=101)  # 101 classes in the dataset

    return image, label

# Convert dataset to TensorFlow dataset
def to_tf_dataset(dataset):
    tf_dataset = tf.data.Dataset.from_generator(
        lambda: ((example['image'], example['label']) for example in dataset),
        output_signature=(
            tf.TensorSpec(shape=IMG_SIZE + (3,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64)
        )
    )
    tf_dataset = tf_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    tf_dataset = tf_dataset.batch(BATCH_SIZE)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    return tf_dataset

# Load datasets
train_ds, test_ds = load_and_split_dataset()
train_tf_dataset = to_tf_dataset(train_ds)
test_tf_dataset = to_tf_dataset(test_ds)

# Create the InceptionV3-based model
input_tensor = Input(shape=(*IMG_SIZE, 3))
base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)

x = base_model.output
x = AveragePooling2D()(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
predictions = Dense(101, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0005), activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
opt = SGD(learning_rate=0.1, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpointer = ModelCheckpoint(filepath='recipe_image_classification_model.h5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('training_log.csv')
lr_scheduler = LearningRateScheduler(schedule)

# Plotting function for accuracy and loss
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the figure
    plt.savefig('training_accuracy_loss.png')
    plt.show()

# Training the model with detailed logging
class DetailedLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        print(f"Epoch {epoch + 1}/{EPOCHS} begins")

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        print(f"Epoch {epoch + 1}/{EPOCHS} completed in {epoch_time:.2f} seconds")
        print(f"Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, "
              f"Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}")

# Train the model
history = model.fit(
    train_tf_dataset,
    validation_data=test_tf_dataset,
    epochs=EPOCHS,
    callbacks=[checkpointer, csv_logger, lr_scheduler, DetailedLoggingCallback()]
)

# Plot and save the accuracy and loss
plot_training_history(history)

# Save the final model
model.save('recipe_image_classification_model.h5')

# Log the successful completion of the training process
print("Model is trained successfully!")

# Test with one image from the test dataset and log the result
for image_batch, label_batch in test_tf_dataset.take(1):
    predictions = model.predict(image_batch)
    predicted_label = np.argmax(predictions[0])
    true_label = np.argmax(label_batch[0])

    print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
    break

