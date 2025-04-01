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
from scipy.signal import resample
import pretty_midi
import json
from tqdm import tqdm
import multiprocessing
import time
import gc
import warnings
import functools  # For caching
from numba import jit, prange  # For JIT compilation and parallel loops
import threading

# Try to import optional acceleration libraries
try:
    import numba
    HAS_NUMBA = True
    print("Numba JIT acceleration available")
except ImportError:
    HAS_NUMBA = False
    print("Numba JIT acceleration not available")

# ## 1. Configuration


# Data paths
# Update these paths to point to the actual location of your dataset
RAW_DATA_PATH = "data/raw/maestro-v3.0.0"
PROCESSED_DATA_PATH = "data/processed/maestro-v3.0.0"

# Create processed data directory if it doesn't exist
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# Audio parameters
SR = 22050  # Sample rate in Hz
DURATION = 0.1 # Window size in seconds
HOP_LENGTH = 512  # Hop length for STFT
N_FFT = 2048  # Number of FFT points
N_MELS = 128  # Number of mel bands
FMIN = 27.5  # Lowest piano key frequency (A0)
FMAX = 4186.0  # Highest piano key frequency (C8)

# Performance optimization settings
BATCH_SIZE = 200  # Increased batch size
PARALLEL_WINDOWS = True  # Process windows in parallel
USE_MEMMAP = True  # Use memory mapping for large arrays
USE_JIT = HAS_NUMBA  # Use numba JIT acceleration if available
CACHE_ENABLED = True  # Cache expensive function results
N_THREADS = max(1, multiprocessing.cpu_count() - 1)  # Save one core for system

# Caching decorator
def mem_cache(func):
    """Simple in-memory cache for function results"""
    if not CACHE_ENABLED:
        return func
    
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper

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

@mem_cache
def get_mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """
    Get or create a mel filterbank matrix for faster processing (cached)
    """
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

def compute_mel_spectrogram_optimized(y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, 
                                   hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX):
    """
    Optimized mel spectrogram computation
    """
    # Get the precomputed mel filterbank
    mel_fb = get_mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    
    # Compute power spectrogram
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    
    # Apply mel filterbank (faster than recomputing it every time)
    return np.dot(mel_fb, D)

compute_mel_spectrogram = compute_mel_spectrogram_optimized

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
    Compute harmonic product spectrum

    """
    H = X.copy()
    for n in range(2, n_harmonics + 1):
        downsampled = resample(X, len(X) // n, axis=0)
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
        if instrument.program >= 8:
            continue
        for note in instrument.notes:
            notes.append({
                'pitch': note.pitch,
                'start': note.start,
                'end': note.end,
                'velocity': note.velocity
            })

    return sorted(notes, key=lambda x: x['start'])

def create_note_labels_optimized(notes, timestamps, duration):
    """
    Optimized version of note label creation using vectorization
    """
    # 88 piano keys (21-108 MIDI pitch)
    n_keys = 88
    n_timestamps = len(timestamps)
    
    # Create empty label matrix
    labels = np.zeros((n_timestamps, n_keys), dtype=np.uint8)  # Use uint8 to save memory
    
    if not notes:
        return labels
    
    # Convert notes to numpy arrays for faster processing
    note_pitches = np.array([note['pitch'] - 21 for note in notes])
    note_starts = np.array([note['start'] for note in notes])
    note_ends = np.array([note['end'] for note in notes])
    
    # Check which notes are valid piano keys (0-87)
    valid_notes = (note_pitches >= 0) & (note_pitches < n_keys)
    note_pitches = note_pitches[valid_notes]
    note_starts = note_starts[valid_notes]
    note_ends = note_ends[valid_notes]
    
    # For each timestamp
    end_times = timestamps + duration
    
    for i in range(n_timestamps):
        # Find notes that overlap with this window
        # A note overlaps if: note_end > timestamp_start AND note_start < timestamp_end
        overlapping = (note_ends > timestamps[i]) & (note_starts < end_times[i])
        
        # Set the active notes to 1
        labels[i, note_pitches[overlapping]] = 1
    
    return labels

create_note_labels = create_note_labels_optimized


# ## 6. Preprocessing Pipeline


def process_windows_batch(windows_batch, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Process a batch of windows using threading for I/O bound operations
    """
    mel_fb = get_mel_filterbank(sr, n_fft, N_MELS, FMIN, FMAX)
    results = []
    
    # Define worker function for each window
    def process_window(window, idx):
        mel_spec = compute_mel_spectrogram(window, sr, N_MELS, n_fft, hop_length, FMIN, FMAX)
        results.append((idx, mel_spec))
    
    # Use threading for I/O bound operations
    threads = []
    for idx, window in enumerate(windows_batch):
        thread = threading.Thread(target=process_window, args=(window, idx))
        threads.append(thread)
        thread.start()
        
        # Limit concurrent threads to avoid overwhelming the system
        if len(threads) >= N_THREADS:
            for t in threads:
                t.join()
            threads = []
    
    # Wait for remaining threads
    for t in threads:
        t.join()
    
    # Sort results by index and extract features
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]

def preprocess_sample(audio_path, midi_path, sample_id, output_dir, sr=SR, duration=DURATION):
    """
    Optimized preprocessing for a single audio-MIDI pair
    """
    # Start timing
    start_time = time.time()
    
    # Create output directories
    features_dir = os.path.join(output_dir, 'features')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Load audio - this is I/O bound so can't optimize much
    y = load_audio(audio_path, sr=sr)
    y = normalize_audio(y)
    
    # Load MIDI
    midi = load_midi(midi_path)
    notes = extract_piano_notes(midi)
    
    # Extract windows efficiently
    windows, timestamps = extract_windows(y, sr=sr, duration=duration)
    
    # Batch size for processing windows
    window_batch_size = 200
    window_limit = 1000  # For demonstration purposes
    windows_to_process = min(len(windows), window_limit)
    
    # Process in batches - use disable=True to hide this inner progress bar
    for batch_idx in range(0, windows_to_process, window_batch_size):
        # Get current batch
        end_idx = min(batch_idx + window_batch_size, windows_to_process)
        window_batch = windows[batch_idx:end_idx]
        timestamps_batch = timestamps[batch_idx:end_idx]
        
        # Create labels once per batch (more efficient)
        labels_batch = create_note_labels(notes, timestamps_batch, duration)
        
        # Process all windows in this batch with optimized function
        features_batch = process_windows_batch(window_batch, sr)
        
        # Save batch results efficiently
        for i in range(len(window_batch)):
            abs_idx = batch_idx + i
            window_id = f"{sample_id}_{abs_idx:06d}"
            
            # Save features - minimize I/O by batching
            feature_path = os.path.join(features_dir, f"{window_id}.npz")
            np.savez_compressed(feature_path, mel_spectrogram=features_batch[i])
            
            # Save label
            label_path = os.path.join(labels_dir, f"{window_id}.npy")
            np.save(label_path, labels_batch[i])
        
        # Clean up to avoid memory leaks
        del features_batch, window_batch, labels_batch
        gc.collect()
    
    # Save metadata
    metadata = {
        'sample_id': sample_id,
        'audio_path': audio_path,
        'midi_path': midi_path,
        'duration': len(y) / sr,
        'n_windows': windows_to_process,
        'window_duration': duration,
        'sr': sr,
        'processing_time': time.time() - start_time
    }
    
    metadata_path = os.path.join(output_dir, f"{sample_id}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def process_split(split_name, split_df, raw_data_path, processed_data_path, processed_metadata_queue, log_queue):
    """
    Processes a single split of the MAESTRO dataset.
    """
    split_output_dir = os.path.join(processed_data_path, split_name)
    os.makedirs(split_output_dir, exist_ok=True)

    # Log the start of processing for this split
    log_queue.put(f"Starting processing for split: {split_name} with {len(split_df)} samples")
    
    split_metadata = []
    # Set position=split_index to avoid overlap, leave=False to clean up after completion
    for idx, row in enumerate(tqdm(split_df.iterrows(), total=len(split_df), 
                                   desc=f"Processing {split_name}", position=0, leave=True,
                                   bar_format='{l_bar}{bar:30}{r_bar}')):  # More compact progress bar
        _, row = row  # Unpack the tuple
        sample_id = row['canonical_composer'] + '_' + os.path.basename(row['midi_filename']).replace('.midi', '')
        audio_path = os.path.join(raw_data_path, row['audio_filename'])
        midi_path = os.path.join(raw_data_path, row['midi_filename'])

        # Skip if files don't exist
        if not os.path.exists(audio_path) or not os.path.exists(midi_path):
            tqdm.write(f"Skipping {sample_id} - files not found")  # Use tqdm.write instead of print
            continue

        # Use tqdm.write for console output within loops to avoid progress bar issues
        tqdm.write(f"Processing {idx+1}/{len(split_df)} in {split_name}: {sample_id}")
        log_queue.put(f"Processing {idx+1}/{len(split_df)} in {split_name}: {sample_id}")
        
        try:
            metadata = preprocess_sample(audio_path, midi_path, sample_id, split_output_dir)
            split_metadata.append(metadata)
            msg = f"Completed {sample_id} - processed {metadata['n_windows']} windows"
            tqdm.write(msg)  # Display in console in a tqdm-friendly way
        except Exception as e:
            log_msg = f"Error processing {sample_id} in split {split_name}: {e}"
            tqdm.write(log_msg)  # Use tqdm.write instead of print
    
    log_queue.put(f"Finished processing split: {split_name}")
    processed_metadata_queue.put(split_metadata)


# Move the log handler function to module level
def log_handler(log_queue):
    """Process log messages from the queue"""
    while True:
        message = log_queue.get()
        if message == "DONE":
            break
        # For log handler, we don't use print to avoid cluttering the console with duplicate messages
        # This log queue can be used for logging to a file instead

# ## 7. Process a Subset of MAESTRO

if __name__ == '__main__':
    # Add multiprocessing safeguards for Windows
    if os.name == 'nt':  # Windows
        # Set start method to 'spawn' (default on Windows, but being explicit)
        multiprocessing.set_start_method('spawn', force=True)

    # Adjust SAMPLE_COUNT or remove the limit for full dataset processing
    SAMPLE_COUNT = 10  # Number of samples to process from each split
    NUM_PROCESSES = multiprocessing.cpu_count()  # Use all available CPU cores
    
    # Add option to disable progress bars
    DISABLE_PROGRESS_BARS = False  # Set to True to disable all progress bars

    # Get a few samples from each split
    samples = {}
    for split in ['train', 'validation', 'test']:
        split_df = maestro_df[maestro_df['split'] == split].head(SAMPLE_COUNT)
        samples[split] = split_df

    processed_metadata = []
    processed_metadata_queue = multiprocessing.Queue()
    log_queue = multiprocessing.Queue()  # Queue for collecting log messages
    processes = []

    # Start a separate process to handle logging
    log_thread = multiprocessing.Process(
        target=log_handler,
        args=(log_queue,)
    )
    log_thread.start()

    # Start all processing jobs
    print(f"Starting processing with {NUM_PROCESSES} processes")
    total_samples = sum(len(df) for df in samples.values())
    print(f"Total samples to process: {total_samples}")
    
    # More concise status reporting
    for split, split_df in samples.items():
        # Make sure all arguments are pickle-compatible
        process = multiprocessing.Process(
            target=process_split,
            args=(split, split_df, RAW_DATA_PATH, PROCESSED_DATA_PATH, processed_metadata_queue, log_queue)
        )
        processes.append(process)
        process.start()
        print(f"Started process for {split} split")

    # Wait for all processes to complete
    for process in processes:
        process.join()
    
    # Signal log thread to finish
    log_queue.put("DONE")
    log_thread.join()

    print("All processing complete! Collecting metadata...")
    
    # Collect metadata from the queue
    while not processed_metadata_queue.empty():
        processed_metadata.extend(processed_metadata_queue.get())
    
    print(f"Collected metadata for {len(processed_metadata)} processed samples")

    # ## 8. Visualize a Sample
    # Visualize a sample (mel spectrogram and labels)
    if processed_metadata:
        print("Attempting to visualize a processed sample...")
        
        # Try to find a valid sample to visualize
        valid_sample_found = False
        sample_to_visualize = None
        
        for metadata_item in processed_metadata:
            sample_id = metadata_item['sample_id']
            for split_name in samples.keys():
                split_output_dir = os.path.join(PROCESSED_DATA_PATH, split_name)
                
                # Get the first window
                window_id = f"{sample_id}_000000"
                feature_path = os.path.join(split_output_dir, 'features', f"{window_id}.npz")
                label_path = os.path.join(split_output_dir, 'labels', f"{window_id}.npy")
                
                # Check if both files exist
                if os.path.exists(feature_path) and os.path.exists(label_path):
                    print(f"Found valid sample to visualize: {sample_id} in {split_name}")
                    sample_to_visualize = {
                        'sample_id': sample_id,
                        'split': split_name,
                        'feature_path': feature_path,
                        'label_path': label_path
                    }
                    valid_sample_found = True
                    break
            
            if valid_sample_found:
                break
                
        if valid_sample_found:
            try:
                # Load features and label
                features = np.load(sample_to_visualize['feature_path'])
                mel_spec = features['mel_spectrogram']
                label = np.load(sample_to_visualize['label_path'])
                
                # Plot mel spectrogram
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 1, 1)
                librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max),
                                         y_axis='mel', x_axis='time', sr=SR, fmin=FMIN, fmax=FMAX)
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Mel spectrogram - {sample_to_visualize["sample_id"]}')
                
                # Plot piano roll
                plt.subplot(2, 1, 2)
                plt.imshow(label.reshape(1, -1), aspect='auto', interpolation='nearest', cmap='Blues')
                plt.yticks([])
                plt.xlabel('Piano Key (A0-C8)')
                plt.title('Active Notes')
                
                plt.tight_layout()
                plt.show()
                
                print("Visualization complete!")
            except Exception as e:
                print(f"Error during visualization: {e}")
        else:
            print("No valid samples found for visualization. This could mean that:")
            print("1. No samples were successfully processed")
            print("2. The processed files are not in the expected locations")
            print("3. There were issues with file permissions or paths")
            
    else:
        print("No metadata available for visualization - no samples were successfully processed")

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
    # After preprocessing, you can:
    # 1. Load the processed data into TensorFlow Dataset objects
    # 2. Build and train your model
    # 3. Evaluate performance on test set
    # 4. Optimize for latency
    # For full dataset processing:
    # - Remove the sample limit (SAMPLE_COUNT)
    # - Adjust the window limit in preprocess_sample function
    # - Consider using parallel processing to speed up preprocessing
    # - Depending on the file size constraints, maybe use sharding or distributed processing
