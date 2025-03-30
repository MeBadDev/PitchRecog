# PitchRecog - Realtime Piano Note Recognition

## Project Overview
PitchRecog is focused on training a machine learning model for realtime piano note recognition. This repository contains the model development process, with the end goal of creating a model that can be deployed to web applications. The system aims to identify piano notes from audio input with minimal latency (≤50ms).

## Technical Stack
- **Model Development**: Python with TensorFlow 
- **Audio Processing**: Librosa/PyAudio for audio analysis
- **Data Handling**: NumPy, Pandas for dataset management
- **Future Deployment**: Model will be exported for web deployment (via TensorFlow.js)

## Key Features
- Real-time piano note detection
- Low latency processing (≤50ms window)
- Web-based interface
- Support for polyphonic piano sounds (multiple simultaneous notes)

## Performance Requirements
- Maximum processing latency: 50ms
- High accuracy for standard piano notes (A0-C8)
- Robust to background noise and recording conditions

## Development Roadmap
1. Data collection/preparation of piano note samples
2. Audio preprocessing and feature extraction
3. Model architecture design and implementation
4. Training and hyperparameter optimization
5. Model evaluation and validation
6. Optimization for low-latency inference
7. Export model for web deployment

## Audio Preprocessing Pipeline

For optimal model training and performance, the following preprocessing steps are recommended:

### 1. Audio Data Preprocessing
- **Sample Rate Conversion**: Standardize all audio to 44.1kHz or 22.05kHz
- **Audio Windowing**: Create fixed-length segments (20-50ms) with potential overlap
- **Amplitude Normalization**: Scale audio to consistent levels to handle volume variations

### 2. Feature Extraction
- **Time-Frequency Representations**:
  - **Short-Time Fourier Transform (STFT)**: Applies FFT to short, overlapping windows of audio to capture how frequency content changes over time. Creates spectrograms for visual and data representation.
  - **Fast Fourier Transform (FFT)**: Efficient algorithm used within STFT to convert time-domain signals to frequency-domain.
  - **Mel Spectrograms**: Transform to mel scale for better pitch perception alignment
  - **Constant-Q Transform (CQT)**: Better for musical note frequencies as it has logarithmic frequency resolution matching musical scales
- **Music-Specific Features**:
  - **Chromagrams**: For pitch class detection
  - **Harmonic Product Spectrum**: To enhance fundamental frequencies
  - **Onset Detection**: For note boundary identification

### 3. Data Augmentation
- **Pitch Shifting**: Small variations (±1-2 semitones)
- **Time Stretching**: Slight tempo changes (±5-10%)
- **Noise Addition**: Various levels of background noise
- **Room Simulation**: Adding reverb/echo effects

### 4. Dataset Organization
- **Train/Validation/Test Split**: Typically 70/15/15 or 80/10/10
- **Balanced Note Distribution**: Ensure all piano notes are well-represented
- **Polyphonic Considerations**: Include samples with multiple simultaneous notes

### 5. Feature Normalization
- **Standardization**: Zero mean, unit variance
- **Min-Max Scaling**: Scale features to a specific range (e.g., [0,1])

## Web Deployment Performance Considerations

### Latency Analysis
The 50ms latency requirement poses significant challenges for web deployment. Breaking down the timing budget:

- **Audio Capture**: ~5-10ms
- **Preprocessing**: ~10-20ms
- **Model Inference**: ~20-30ms
- **Post-processing & UI Updates**: ~5ms

### Feasibility Assessment
Not all preprocessing techniques described above can be implemented within the 50ms constraint:

- **Feasible within 50ms**:
  - Basic FFT using Web Audio API's AnalyserNode
  - Simple peak detection algorithms
  - Lightweight feature extraction
  - Quantized/optimized neural network inference

- **Likely too computationally expensive**:
  - Full STFT with many overlapping windows
  - Mel spectrogram conversion
  - CQT transformation
  - Chromagram generation
  - Complex harmonic analysis

### Optimization Strategies
To meet the latency requirement:
1. **Train with full features, deploy with minimal features**
   - Train on rich representations but optimize model to work with simpler inputs
   - Use knowledge distillation techniques

2. **Web-optimized implementation**
   - Use Web Audio API's built-in FFT for frequency analysis
   - Implement WebAssembly for critical DSP functions
   - Leverage GPU acceleration via WebGL/WebGPU with TensorFlow.js
   - Consider smaller frame sizes (10-20ms) with less overlap

3. **Model architecture considerations**
   - Quantized 8-bit weights instead of 32-bit float
   - Pruned/compressed neural networks
   - Explore efficient architectures (MobileNet-style, EfficientNet, etc.)

4. **Progressive enhancement**
   - Start with minimal processing and add features based on device capability
   - Implement fallback detection methods for lower-power devices
