# # MAESTRO Dataset Preprocessing
# 
# This notebook preprocesses piano recordings from the MAESTRO dataset (v3.0.0) for piano note recognition model training.
# 
# MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization) dataset contains paired audio and MIDI recordings from piano performances.
# 
# More info: https://magenta.tensorflow.org/datasets/maestro


# Install required libraries


import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pretty_midi
import json
from tqdm import tqdm

# ## 1. Configuration


# Data paths
# Update these paths to point to the actual location of your dataset
RAW_DATA_PATH = "data/raw/maestro-v3.0.0" 
PROCESSED_DATA_PATH = "data/processed/maestro-v3.0.0"

# Create processed data directory if it doesn't exist
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# Audio parameters
SR = 22050  # Sample rate in Hz
DURATION = 0.05  # Window size in seconds (50ms)
HOP_LENGTH = 512  # Hop length for STFT
N_FFT = 2048  # Number of FFT points
N_MELS = 128  # Number of mel bands
FMIN = 27.5  # Lowest piano key frequency (A0)
FMAX = 4186.0  # Highest piano key frequency (C8)

# ## 2. Load Dataset Metadata


# Load the MAESTRO dataset metadata
maestro_csv_path = os.path.join(RAW_DATA_PATH, "maestro-v3.0.0.csv")
maestro_df = pd.read_csv(maestro_csv_path)
maestro_df.head()


# Check data distribution
print(f"Total recordings: {len(maestro_df)}")
print(f"Years: {maestro_df['year'].unique()}")
print(f"Split distribution:\n{maestro_df['split'].value_counts()}")

# ## 3. Audio Preprocessing Functions


def load_audio(audio_path, sr=SR):
    """
    Load audio file with the specified sample rate
    """
    y, _ = librosa.load(audio_path, sr=sr)
    return y

def normalize_audio(y):
    """
    Amplitude normalization to range [-1, 1]
    """
    return librosa.util.normalize(y)

def extract_windows(y, sr=SR, duration=DURATION, hop_duration=None):
    """
    Extract overlapping windows from audio
    """
    window_size = int(sr * duration)
    
    # If hop_duration is not specified, use 50% overlap
    if hop_duration is None:
        hop_duration = duration / 2
    hop_size = int(sr * hop_duration)
    
    # Pad the audio to ensure complete windows
    pad_width = window_size - (len(y) % hop_size)
    if pad_width < window_size:
        y = np.pad(y, (0, pad_width))
    
    windows = []
    timestamps = []
    
    # Extract windows with hop_size step
    for start in range(0, len(y) - window_size + 1, hop_size):
        window = y[start:start + window_size]
        windows.append(window)
        timestamps.append(start / sr)
    
    return np.array(windows), np.array(timestamps)

 
# ## 4. Feature Extraction Functions


def compute_stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Compute Short-Time Fourier Transform
    """
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

def compute_spectrogram(stft):
    """
    Compute power spectrogram from STFT
    """
    return np.abs(stft) ** 2

def compute_mel_spectrogram(y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX):
    """
    Compute mel spectrogram
    """
    return librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmin=fmin, fmax=fmax
    )

def compute_cqt(y, sr=SR, fmin=FMIN):
    """
    Compute Constant-Q Transform
    """
    return np.abs(librosa.cqt(y, sr=sr, fmin=fmin))

def compute_chroma(y, sr=SR, n_chroma=12, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Compute chromagram (pitch class profile)
    """
    return librosa.feature.chroma_stft(
        y=y, sr=sr, n_chroma=n_chroma, n_fft=n_fft, hop_length=hop_length
    )

def compute_harmonic_product_spectrum(X, n_harmonics=5):
    """
    Compute Harmonic Product Spectrum (HPS) to enhance fundamental frequencies
    X: Power spectrogram
    """
    H = X.copy()
    for n in range(2, n_harmonics + 1):
        # Downsample spectrum by factor n
        downsampled = np.zeros_like(X)
        for i in range(len(X) // n):
            downsampled[i] = X[i * n]
        H *= downsampled
    return H

def detect_onsets(y, sr=SR, hop_length=HOP_LENGTH):
    """
    Detect note onsets
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    return onset_times

 
# ## 5. MIDI Processing Functions


def load_midi(midi_path):
    """
    Load MIDI file
    """
    return pretty_midi.PrettyMIDI(midi_path)

def extract_piano_notes(midi):
    """
    Extract piano notes from MIDI file with their onset/offset times
    """
    notes = []
    
    for instrument in midi.instruments:
        # Skip non-piano instruments if any
        if instrument.program >= 8:  # Piano programs are 0-7 in MIDI
            continue
            
        for note in instrument.notes:
            notes.append({
                'pitch': note.pitch,
                'start': note.start,
                'end': note.end,
                'velocity': note.velocity
            })
    
    return sorted(notes, key=lambda x: x['start'])

def create_note_labels(notes, timestamps, duration):
    """
    For each timestamp, create a binary label vector indicating
    which notes are active during the window
    """
    # 88 piano keys (21-108 MIDI pitch)
    n_keys = 88
    
    # Create empty label matrix
    labels = np.zeros((len(timestamps), n_keys))
    
    for i, timestamp in enumerate(timestamps):
        # Window start and end times
        start_time = timestamp
        end_time = timestamp + duration
        
        # Find notes active during this window
        for note in notes:
            # Check if note overlaps with window
            if note['end'] > start_time and note['start'] < end_time:
                # Convert MIDI pitch to piano key index (0-87)
                key_idx = note['pitch'] - 21
                if 0 <= key_idx < n_keys:
                    labels[i, key_idx] = 1
    
    return labels

 
# ## 6. Preprocessing Pipeline


def preprocess_sample(audio_path, midi_path, sample_id, output_dir, sr=SR, duration=DURATION):
    """
    Preprocess a single audio-MIDI pair and save the result
    """
    # Create output directories
    features_dir = os.path.join(output_dir, 'features')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Load and process audio
    y = load_audio(audio_path, sr=sr)
    y = normalize_audio(y)
    
    # Load and process MIDI
    midi = load_midi(midi_path)
    notes = extract_piano_notes(midi)
    
    # Extract overlapping windows and their timestamps
    windows, timestamps = extract_windows(y, sr=sr, duration=duration)
    
    # Create note labels for each window
    labels = create_note_labels(notes, timestamps, duration)
    
    # Store features for each window
    all_features = []
    
    for i, window in enumerate(tqdm(windows, desc=f"Processing {sample_id}", leave=False)):
        # Extract features
        features = {}
        
        # Mel spectrogram (our primary feature)
        mel_spec = compute_mel_spectrogram(window, sr=sr)
        features['mel_spectrogram'] = mel_spec
        
        # CQT (optional)
        # cqt = compute_cqt(window, sr=sr)
        # features['cqt'] = cqt
        
        # Chroma (optional)
        # chroma = compute_chroma(window, sr=sr)
        # features['chroma'] = chroma
        
        # Save each window's features and label to separate files
        window_id = f"{sample_id}_{i:06d}"
        
        # Save features
        feature_path = os.path.join(features_dir, f"{window_id}.npz")
        np.savez_compressed(feature_path, **features)
        
        # Save label
        label_path = os.path.join(labels_dir, f"{window_id}.npy")
        np.save(label_path, labels[i])
        
        # For demonstration, only process a subset of windows
        if i >= 1000:  # Adjust this based on your needs
            break
    
    # Save metadata
    metadata = {
        'sample_id': sample_id,
        'audio_path': audio_path,
        'midi_path': midi_path,
        'duration': len(y) / sr,
        'n_windows': min(len(windows), 1000),  # Adjust based on limit above
        'window_duration': duration,
        'sr': sr
    }
    
    metadata_path = os.path.join(output_dir, f"{sample_id}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

 
# ## 7. Process a Subset of MAESTRO


# Process a small subset for demonstration
# Adjust SAMPLE_COUNT or remove the limit for full dataset processing
SAMPLE_COUNT = 5

# Get a few samples from each split
samples = {}
for split in ['train', 'validation', 'test']:
    split_df = maestro_df[maestro_df['split'] == split].head(SAMPLE_COUNT)
    samples[split] = split_df

processed_metadata = []

for split, split_df in samples.items():
    # Create output directory for this split
    split_output_dir = os.path.join(PROCESSED_DATA_PATH, split)
    os.makedirs(split_output_dir, exist_ok=True)
    
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split} samples"):
        sample_id = row['canonical_composer'] + '_' + os.path.basename(row['midi_filename']).replace('.midi', '')
        audio_path = os.path.join(RAW_DATA_PATH, row['audio_filename'])
        midi_path = os.path.join(RAW_DATA_PATH, row['midi_filename'])
        
        # Skip if files don't exist
        if not os.path.exists(audio_path) or not os.path.exists(midi_path):
            print(f"Skipping {sample_id} - files not found")
            continue
            
        try:
            metadata = preprocess_sample(audio_path, midi_path, sample_id, split_output_dir)
            processed_metadata.append(metadata)
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")

 
# ## 8. Visualize a Sample


# Visualize a sample (mel spectrogram and labels)
if processed_metadata:
    # Get the first processed sample
    sample_id = processed_metadata[0]['sample_id']
    split = list(samples.keys())[0]
    split_output_dir = os.path.join(PROCESSED_DATA_PATH, split)
    
    # Get the first window
    window_id = f"{sample_id}_000000"
    feature_path = os.path.join(split_output_dir, 'features', f"{window_id}.npz")
    label_path = os.path.join(split_output_dir, 'labels', f"{window_id}.npy")
    
    # Load features and label
    features = np.load(feature_path)
    mel_spec = features['mel_spectrogram']
    label = np.load(label_path)
    
    # Plot mel spectrogram
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max),
                           y_axis='mel', x_axis='time', sr=SR, fmin=FMIN, fmax=FMAX)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel spectrogram - {window_id}')
    
    # Plot piano roll
    plt.subplot(2, 1, 2)
    plt.imshow(label.reshape(1, -1), aspect='auto', interpolation='nearest', cmap='Blues')
    plt.yticks([])
    plt.xlabel('Piano Key (A0-C8)')
    plt.title('Active Notes')
    
    plt.tight_layout()
    plt.show()

 
# ## 9. Save Processed Dataset Metadata


# Save overall dataset metadata
dataset_info = {
    'total_samples': len(processed_metadata),
    'sample_rate': SR,
    'window_duration': DURATION,
    'n_mels': N_MELS,
    'fmin': FMIN,
    'fmax': FMAX,
    'processed_samples': processed_metadata
}

with open(os.path.join(PROCESSED_DATA_PATH, 'dataset_info.json'), 'w') as f:
    json.dump(dataset_info, f, indent=2)

 
# ## 10. Next Steps
# 
# After preprocessing, you can:
# 1. Load the processed data into TensorFlow Dataset objects
# 2. Build and train your model
# 3. Evaluate performance on test set
# 4. Optimize for latency
# 
# For full dataset processing:
# - Remove the sample limit (SAMPLE_COUNT)
# - Adjust the window limit in preprocess_sample function
# - Consider using parallel processing to speed up preprocessing
# - Depending on the file size constraints, maybe use sharding or distributed processing


