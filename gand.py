import numpy as np
import os
from data_processing import load_dataset  # Assuming this loads your dataset

DATA_DIR = 'genres'  # Update with your path
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']

def create_filename_array():
    X, y = load_dataset(DATA_DIR, genres)  # Assuming this loads both features and labels

    # Assuming audio files have extensions like .wav or .mp3
    audio_extensions = ('.wav')
    filenames = [f for f in sorted(os.listdir(DATA_DIR)) if f.endswith(audio_extensions)]
    
    if len(filenames) != len(X):
        raise ValueError(f"Mismatch: Found {len(filenames)} filenames but {len(X)} samples in the dataset.")
    
    np.save('processed_filenames.npy', filenames)
    print("Filenames successfully saved as 'processed_filenames.npy'")

if __name__ == "__main__":
    create_filename_array()
