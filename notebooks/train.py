# # MAESTRO Model Training Script
#
# This script trains a model for piano note recognition using the
# preprocessed data from the MAESTRO dataset.

import os
import numpy as np
import tensorflow as tf
from keras import layers, models
import json
import glob
from sklearn.model_selection import train_test_split

# ## 1. Configuration

# Path to the processed data directory (must match your preprocessing script)
PROCESSED_DATA_PATH = "data/processed/maestro-v3.0.0"
# Load dataset info to get parameters used during preprocessing
DATASET_INFO_PATH = os.path.join(PROCESSED_DATA_PATH, 'dataset_info.json')

# Model & Training Parameters
# Input shape depends on the mel spectrogram dimensions from preprocessing
# (n_mels, time_steps_in_mel_spectrogram)
# Let's load one sample to determine the shape
# Adjust BATCH_SIZE and EPOCHS based on your system resources and desired training time
BATCH_SIZE = 128
EPOCHS = 15  # Start with a small number and increase as needed
N_KEYS = 88  # Number of piano keys (output size)

# ## 2. Load Dataset Information & Determine Input Shape

try:
    with open(DATASET_INFO_PATH, 'r') as f:
        dataset_info = json.load(f)
    print("Loaded dataset info:")
    print(f"  Sample Rate: {dataset_info['sample_rate']}")
    print(f"  Window Duration: {dataset_info['window_duration']}")
    print(f"  Num Mel Bands: {dataset_info['n_mels']}")

    # Find the first feature file to determine input shape
    first_feature_file = None
    for split in ['train', 'validation', 'test']:
        split_feature_dir = os.path.join(PROCESSED_DATA_PATH, split, 'features')
        feature_files = glob.glob(os.path.join(split_feature_dir, '*.npz'))
        if feature_files:
            first_feature_file = feature_files[0]
            break

    if not first_feature_file:
        raise FileNotFoundError("Could not find any preprocessed feature files.")

    # Load the first feature file to get the shape
    with np.load(first_feature_file) as data:
        # Assuming 'mel_spectrogram' is the key used in preprocessing
        sample_feature = data['mel_spectrogram']
        INPUT_SHAPE = sample_feature.shape + (1,)  # Add channel dimension for CNN
        print(f"Determined input shape from sample: {INPUT_SHAPE}")

except FileNotFoundError:
    print(f"Error: Could not find dataset_info.json at {DATASET_INFO_PATH}")
    print("Please ensure the preprocessing script ran successfully and the path is correct.")
    exit()
except KeyError as e:
    print(f"Error: Missing key {e} in dataset_info.json or feature file.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading data info: {e}")
    exit()

# ## 3. Data Loading Functions

def get_file_paths(split):
    """Gets lists of feature and corresponding label file paths for a given split."""
    features_dir = os.path.join(PROCESSED_DATA_PATH, split, 'features')
    labels_dir = os.path.join(PROCESSED_DATA_PATH, split, 'labels')

    feature_files = sorted(glob.glob(os.path.join(features_dir, '*.npz')))
    label_files = sorted(glob.glob(os.path.join(labels_dir, '*.npy')))

    # Basic check to ensure corresponding files exist
    if len(feature_files) != len(label_files):
        print(f"Warning: Mismatch in number of feature ({len(feature_files)}) and label ({len(label_files)}) files for split '{split}'.")
        feature_basenames = {os.path.basename(f).replace('.npz', ''): f for f in feature_files}
        label_basenames = {os.path.basename(f).replace('.npy', ''): f for f in label_files}
        common_basenames = sorted(list(feature_basenames.keys() & label_basenames.keys()))
        feature_files = [feature_basenames[b] for b in common_basenames]
        label_files = [label_basenames[b] for b in common_basenames]
        print(f"Aligned to {len(feature_files)} common files.")

    if not feature_files:
         print(f"Warning: No feature files found for split '{split}' at {features_dir}")
    if not label_files:
         print(f"Warning: No label files found for split '{split}' at {labels_dir}")

    return feature_files, label_files

def load_data(feature_path, label_path):
    """Loads a single feature (mel spectrogram) and label pair."""
    # Load feature (assuming it's saved under 'mel_spectrogram' key)
    feature_data = np.load(feature_path.numpy().decode('utf-8'))
    feature = feature_data['mel_spectrogram'].astype(np.float32)
    # Add channel dimension
    feature = np.expand_dims(feature, axis=-1)

    # Load label
    label = np.load(label_path.numpy().decode('utf-8')).astype(np.float32)
    return feature, label

def create_dataset(feature_files, label_files, batch_size):
    """Creates a tf.data.Dataset for training or validation."""
    if not feature_files or not label_files:
        return None  # Return None if lists are empty

    dataset = tf.data.Dataset.from_tensor_slices((feature_files, label_files))
    dataset = dataset.shuffle(len(feature_files))  # Shuffle files
    # Use tf.py_function to wrap the numpy loading functions
    dataset = dataset.map(lambda feat_path, lab_path: tf.py_function(
        load_data, [feat_path, lab_path], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.AUTOTUNE)

    # Set shapes after loading (important for model compatibility)
    dataset = dataset.map(lambda feature, label: (tf.ensure_shape(feature, INPUT_SHAPE), tf.ensure_shape(label, (N_KEYS,))))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Optimize performance
    return dataset

# ## 4. Define the Model Architecture (Simple CNN Example)

def build_model(input_shape, num_classes):
    """Builds a CNN model adapted for short time dimension."""
    model = models.Sequential(name="PianoNoteCNN_Adapted")

    # Use Input layer to define shape explicitly (Good practice)
    model.add(layers.Input(shape=input_shape))

    # Convolutional layers with 'same' padding
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 1)))  # Pool along frequency axis

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 1)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 1)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    return model

# ## 5. Define Custom Weighted Loss Function

def weighted_binary_crossentropy(pos_weights):
    """
    pos_weights: a NumPy array or tensor of shape (num_classes,) representing
                 the weight for each class when the true label is 1.
    Returns a loss function that applies weighted binary crossentropy.
    """
    pos_weights = tf.constant(pos_weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        # Compute standard binary crossentropy for each element
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        # Weight positive labels using pos_weights; negatives remain 1
        weight_matrix = y_true * pos_weights + (1 - y_true)
        weighted_bce = bce * weight_matrix
        return tf.reduce_mean(weighted_bce)
    return loss

# ## 6. Load Data Splits & Compute Per-Class Weights

print("\nLoading data splits...")
train_features, train_labels = get_file_paths('train')
val_features, val_labels = get_file_paths('validation')
test_features, test_labels = get_file_paths('test')

print(f"Found {len(train_features)} training samples.")
print(f"Found {len(val_features)} validation samples.")
print(f"Found {len(test_features)} test samples.")

# Create tf.data datasets
train_dataset = create_dataset(train_features, train_labels, BATCH_SIZE)
val_dataset = create_dataset(val_features, val_labels, BATCH_SIZE)
test_dataset = create_dataset(test_features, test_labels, BATCH_SIZE)

# Load all training labels and stack them into one array (num_samples, 88)
all_labels = []
for label_path in train_labels:
    label = np.load(label_path)
    all_labels.append(label)
all_labels = np.array(all_labels)

# Compute positive and negative counts per piano key
positive_counts = np.sum(all_labels, axis=0)
negative_counts = all_labels.shape[0] - positive_counts

# Compute per-class positive weights
epsilon = 1e-7  # to avoid division by zero
# Compute raw weights
raw_weights = negative_counts / (positive_counts + epsilon)
# Clip the weights to a maximum value (e.g., 10)
pos_weights = np.clip(raw_weights, a_min=1.0, a_max=10.0)
print("Per-class positive weights:", pos_weights)

if train_dataset is None:
    print("\nError: No training data loaded. Cannot proceed with training.")
    exit()
if val_dataset is None:
    print("\nWarning: No validation data loaded. Proceeding without validation split during training.")

# ## 7. Build & Compile the Model with Custom Loss

model = build_model(INPUT_SHAPE, N_KEYS)

# Create custom loss using computed per-class positive weights
custom_loss = weighted_binary_crossentropy(pos_weights)

model.compile(optimizer='adam',
              loss=custom_loss,  # Using custom weighted loss
              metrics=[tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.AUC(name='auc')])

model.summary()

# ## 8. Train the Model

print("\nStarting training...")

# Optional: Add callbacks like EarlyStopping or ModelCheckpoint
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    # tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
]

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks if val_dataset else []
)
print("\nTraining finished.")

# ## 9. Evaluate the Model (Optional)

if test_dataset:
    print("\nEvaluating on test set...")
    test_loss, test_precision, test_recall, test_auc = model.evaluate(test_dataset)
    print(f"\nTest Loss: {test_loss}")
    print(f"Test Precision: {test_precision}")
    print(f"Test Recall: {test_recall}")
    print(f"Test AUC: {test_auc}")
else:
    print("\nNo test data found. Skipping final evaluation.")

# ## 10. Save the Model (Optional)

print("\nSaving model...")
model.save('piano_note_recognition_model.keras')
print("Model saved as piano_note_recognition_model.keras")

# ## 11. Plot Training History (Optional)

import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Plot Precision and Recall
    plt.subplot(1, 2, 1)
    plt.plot(history.history['precision'], label='Train Precision')
    if 'val_precision' in history.history:
        plt.plot(history.history['val_precision'], label='Val Precision')
    plt.plot(history.history['recall'], label='Train Recall')
    if 'val_recall' in history.history:
        plt.plot(history.history['val_recall'], label='Val Recall')
    plt.title('Model Precision and Recall')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    # Plot Loss and AUC
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.plot(history.history['auc'], label='Train AUC')
    if 'val_auc' in history.history:
        plt.plot(history.history['val_auc'], label='Val AUC')
    plt.title('Model Loss and AUC')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if history:
   plot_history(history)
