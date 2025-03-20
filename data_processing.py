import numpy as np
import os
import wave
import struct
from scipy.io import wavfile
from scipy.signal import stft
import librosa
import matplotlib.pyplot as plt

def load_audio(file_path):
    """Load audio file and return sample rate and data."""
    sample_rate, data = wavfile.read(file_path)

    # Ensure data is converted to float32 format
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max  # Normalize integer data to float32

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    return sample_rate, data

def create_spectrogram(audio_data, sample_rate, n_fft=2048, hop_length=512):
    """Create a spectrogram from audio data."""
    frequencies, times, Zxx = stft(audio_data, fs=sample_rate, nperseg=n_fft, noverlap=n_fft-hop_length)
    spectrogram = np.abs(Zxx)  # Magnitude spectrum
    return spectrogram, frequencies, times

def mel_filterbank(num_mel_filters=256, fft_size=2048, sample_rate=22050):
    """Create mel filterbank for mel spectrogram conversion."""
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    low_freq_mel = hz_to_mel(0)
    high_freq_mel = hz_to_mel(sample_rate // 2)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_mel_filters + 2)
    hz_points = np.array([mel_to_hz(mel) for mel in mel_points])
    
    bin_numbers = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    
    fbank = np.zeros((num_mel_filters, fft_size // 2 + 1))
    for i in range(1, num_mel_filters + 1):
        f_m_minus, f_m, f_m_plus = bin_numbers[i-1], bin_numbers[i], bin_numbers[i+1]

        for k in range(f_m_minus, f_m):
            fbank[i-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbank[i-1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return fbank

def mel_spectrogram(spectrogram, sample_rate, num_mel_filters=256, fft_size=2048):
    """Convert spectrogram to mel spectrogram."""
    mel_fb = mel_filterbank(num_mel_filters=num_mel_filters, fft_size=fft_size, sample_rate=sample_rate)
    mel_spec = np.dot(mel_fb, spectrogram)
    mel_spec_db = 10 * np.log10(mel_spec + 1e-10)  # Convert to dB scale
    return mel_spec_db

def augment_audio(audio_data, sample_rate):
    """Add data augmentation for audio robustness."""
    augmented_data = [
        librosa.effects.time_stretch(audio_data, rate=0.9),    # Slow down
        librosa.effects.time_stretch(audio_data, rate=1.1),    # Speed up
        audio_data + 0.005 * np.random.randn(len(audio_data))  # Add noise
    ]
    return augmented_data

def extract_features(file_path, max_len=260):
    try:
        sample_rate, audio_data = load_audio(file_path)
        spectrogram, frequencies, times = create_spectrogram(audio_data, sample_rate)
        mel_spec = mel_spectrogram(spectrogram, sample_rate)

        if mel_spec.size == 0:  # 🚨 Empty data check
            print(f"Warning: Empty features extracted from {file_path}")
            return None

        # Normalize
        mel_spec = (mel_spec - np.mean(mel_spec)) / (np.std(mel_spec) + 1e-10)

        if mel_spec.shape[1] > max_len:
            mel_spec = mel_spec[:, :max_len]
        else:
            pad_width = max_len - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')

        return mel_spec
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

FEATURES_FILE = 'processed_features.npy'
LABELS_FILE = 'processed_labels.npy'
FILENAMES_FILE = 'file_names.npy'

def load_dataset(data_dir, genres, max_samples_per_genre=100):
    """Load dataset from directory structure."""
    if os.path.exists(FEATURES_FILE) and os.path.exists(LABELS_FILE):
        print("Loading preprocessed features...")
        X = np.load(FEATURES_FILE)
        y = np.load(LABELS_FILE)
        filenames = np.load(FILENAMES_FILE)
        return X, y, filenames

    print("Extracting and saving features...")
    X = []
    y = []
    filenames = [] 
    
    for i, genre in enumerate(genres):
        genre_dir = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_dir):
            continue
            
        audio_files = os.listdir(genre_dir)
        count = 0
        
        for audio_file in audio_files:
            if count >= max_samples_per_genre:
                break
                
            if audio_file.endswith('.wav'):
                file_path = os.path.join(genre_dir, audio_file)
                features = extract_features(file_path)
                
                if features is not None:
                    for feature in features:  # Handle augmented data
                        X.append(feature)
                        y.append(i)
                        filenames.append(audio_file)
                    count += 1
    
    print(filenames[:2])

    # Convert to numpy array and reshape to 3D (samples, time_steps, features)
    X = np.array(X)
    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=1)

    # Save processed data
    np.save(FEATURES_FILE, X)
    np.save(LABELS_FILE, np.array(y))
    np.save(FILENAMES_FILE, np.array(filenames))

    return X, np.array(y), np.array(filenames)
