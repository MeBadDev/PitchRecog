"""
MAESTRO Dataset Recovery Script

This script recovers from a failed preprocessing run by:
1. Scanning the processed data directories
2. Validating each feature and label file
3. Removing corrupted files
4. Regenerating the dataset_info.json file
"""

import os
import json
import numpy as np
from tqdm import tqdm
import glob
import sys

# Configuration - ensure these match your original preprocessing
PROCESSED_DATA_PATH = "data/processed/maestro-v3.0.0"
SPLITS = ["train", "validation", "test"]
SR = 22050  # Sample rate in Hz
DURATION = 0.1  # Window size in seconds
N_MELS = 128  # Number of mel bands
FMIN = 27.5  # Lowest piano key frequency (A0)
FMAX = 4186.0  # Highest piano key frequency (C8)

def validate_feature_file(file_path):
    """Validate a feature file (.npz) containing mel spectrogram data"""
    try:
        with np.load(file_path) as data:
            # Check if 'mel_spectrogram' key exists and has correct dimensions
            if 'mel_spectrogram' not in data:
                return False, "Missing mel_spectrogram"
            
            mel_spec = data['mel_spectrogram']
            if mel_spec.ndim != 2:
                return False, f"Incorrect dimension for mel spectrogram: {mel_spec.ndim}"
            
            if mel_spec.shape[0] != N_MELS:
                return False, f"Incorrect number of mel bands: {mel_spec.shape[0]}"
            
            return True, None
    except Exception as e:
        return False, f"Error loading file: {str(e)}"

def validate_label_file(file_path):
    """Validate a label file (.npy) containing piano note activations"""
    try:
        label = np.load(file_path)
        
        # Piano labels should be a vector of 88 elements (88 piano keys)
        if label.ndim != 1 or label.shape[0] != 88:
            return False, f"Incorrect label shape: {label.shape}"
        
        # Values should be binary (0 or 1)
        if not np.all(np.logical_or(label == 0, label == 1)):
            return False, f"Non-binary values in label"
        
        return True, None
    except Exception as e:
        return False, f"Error loading file: {str(e)}"

def find_corresponding_files(feature_path):
    """Find the corresponding label file for a feature file"""
    base_dir = os.path.dirname(os.path.dirname(feature_path))
    file_name = os.path.basename(feature_path).replace('.npz', '.npy')
    label_path = os.path.join(base_dir, 'labels', file_name)
    
    # Extract sample_id from filename (format: sample_id_XXXXXX.npz)
    parts = os.path.basename(feature_path).split('_')
    window_id = parts[-1].replace('.npz', '')
    sample_id = '_'.join(parts[:-1])
    metadata_path = os.path.join(base_dir, f"{sample_id}_metadata.json")
    
    return label_path, metadata_path, sample_id, window_id

def validate_sample_pair(feature_path, label_path):
    """Validate a feature-label file pair"""
    # Validate feature file
    feature_valid, feature_error = validate_feature_file(feature_path)
    if not feature_valid:
        return False, f"Feature file error: {feature_error}"
    
    # Validate label file
    if not os.path.exists(label_path):
        return False, f"Missing label file: {label_path}"
    
    label_valid, label_error = validate_label_file(label_path)
    if not label_valid:
        return False, f"Label file error: {label_error}"
    
    return True, None

def remove_file(file_path):
    """Safely remove a file and log the action"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
    except Exception as e:
        print(f"Failed to remove {file_path}: {e}")
        return False
    return False

def scan_and_validate():
    """Scan all processed directories, validate files, and collect metadata"""
    valid_samples = []
    corrupted_files = []
    sample_metadata = {}
    window_counts = {}
    
    print("Scanning processed data directories...")
    
    # Process each split
    for split in SPLITS:
        split_dir = os.path.join(PROCESSED_DATA_PATH, split)
        if not os.path.exists(split_dir):
            print(f"Warning: Split directory {split_dir} not found, skipping.")
            continue
        
        # Find all feature files in this split
        features_dir = os.path.join(split_dir, 'features')
        if not os.path.exists(features_dir):
            print(f"Warning: Features directory {features_dir} not found, skipping.")
            continue
        
        feature_files = glob.glob(os.path.join(features_dir, '*.npz'))
        
        print(f"Found {len(feature_files)} feature files in {split} split")
        
        # Check each feature file and its corresponding label file
        for feature_path in tqdm(feature_files, desc=f"Validating {split}"):
            label_path, metadata_path, sample_id, window_id = find_corresponding_files(feature_path)
            
            # Validate files
            valid, error = validate_sample_pair(feature_path, label_path)
            
            if valid:
                # Track valid samples and their window counts
                window_key = (split, sample_id)
                if window_key not in window_counts:
                    window_counts[window_key] = 0
                window_counts[window_key] += 1
                
                # Add to valid samples
                valid_samples.append({
                    'split': split,
                    'sample_id': sample_id,
                    'window_id': window_id,
                    'feature_path': feature_path,
                    'label_path': label_path
                })
            else:
                # Add to corrupted files
                corrupted_files.append({
                    'feature_path': feature_path,
                    'label_path': label_path,
                    'error': error
                })
                
                # Remove corrupted files
                remove_file(feature_path)
                remove_file(label_path)
                
            # Load metadata if it exists (once per sample)
            if sample_id not in sample_metadata and os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        sample_metadata[sample_id] = metadata
                except Exception as e:
                    print(f"Error loading metadata {metadata_path}: {e}")
    
    print(f"Validation complete. Found {len(valid_samples)} valid windows and {len(corrupted_files)} corrupted files.")
    
    # Build processed_metadata with accurate window counts
    processed_metadata = []
    for sample_id, metadata in sample_metadata.items():
        # Find all splits where this sample appears
        for split in SPLITS:
            if (split, sample_id) in window_counts:
                # Create a copy of the metadata with updated window count
                updated_metadata = metadata.copy()
                updated_metadata['n_windows'] = window_counts[(split, sample_id)]
                updated_metadata['split'] = split
                processed_metadata.append(updated_metadata)
    
    return processed_metadata, valid_samples, corrupted_files

def generate_dataset_info(processed_metadata):
    """Generate the dataset_info.json file"""
    dataset_info = {
        'total_samples': len(processed_metadata),
        'sample_rate': SR,
        'window_duration': DURATION,
        'n_mels': N_MELS,
        'fmin': FMIN,
        'fmax': FMAX,
        'processed_samples': processed_metadata
    }
    
    # Save the dataset info
    output_path = os.path.join(PROCESSED_DATA_PATH, 'dataset_info.json')
    with open(output_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Generated dataset_info.json with {len(processed_metadata)} samples")
    return output_path

def main():
    """Main function to run the recovery process"""
    print("Starting recovery process for MAESTRO dataset...")
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Processed data path {PROCESSED_DATA_PATH} not found.")
        return
    
    # Scan and validate all files
    processed_metadata, valid_samples, corrupted_files = scan_and_validate()
    
    # Print summary
    total_windows = len(valid_samples)
    total_samples = len(processed_metadata)
    total_corrupted = len(corrupted_files)
    
    print("\nRecovery Summary:")
    print(f"Total valid samples: {total_samples}")
    print(f"Total valid windows: {total_windows}")
    print(f"Total corrupted files removed: {total_corrupted}")
    
    # Generate dataset_info.json
    if processed_metadata:
        output_path = generate_dataset_info(processed_metadata)
        print(f"Recovery complete! Dataset info saved to: {output_path}")
    else:
        print("Error: No valid metadata found. Could not generate dataset_info.json")
    
    # Save corruption report if any corrupted files
    if corrupted_files:
        report_path = os.path.join(PROCESSED_DATA_PATH, 'corruption_report.json')
        with open(report_path, 'w') as f:
            json.dump(corrupted_files, f, indent=2)
        print(f"Corruption report saved to: {report_path}")

if __name__ == "__main__":
    main()
