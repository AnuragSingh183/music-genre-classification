import numpy as np
import os
import joblib
from data_processing import load_dataset
from new_evaluate import evaluate_model, train_test_split
from rnn_cells import RNNModel

# Load saved models
MODEL_DIR = 'saved_models'

def load_model(model_name):
    """Load a saved model."""
    model_file = os.path.join(MODEL_DIR, model_name)
    return joblib.load(model_file)

def test_models(X_test, y_test, genres):
    model_types = ['vanilla']

    for model_type in model_types:
        print(f"\nTesting {model_type.upper()} model on 20 samples...")

        # Load the saved model
        model = load_model(f"{model_type}_model.joblib")

        # Select 20 samples from each genre
        samples_idx = []
        for genre in range(len(genres)):
            idx = np.where(y_test == genre)[0][:20]
            samples_idx.extend(idx)
        
        X_test_samples = X_test[samples_idx]
        y_test_samples = y_test[samples_idx]

        # Evaluate
        predictions = model.predict(X_test_samples)
        
        # Display sample results
        print(f"\n{'Sample Index':<15}{'Actual Genre':<20}{'Predicted Genre'}")
        print("-" * 55)

        for i, sample_idx in enumerate(samples_idx):
            actual_genre = genres[y_test[sample_idx]]
            predicted_genre = genres[predictions[i]]
            print(f"{sample_idx:<15}{actual_genre:<20}{predicted_genre}")

        # Show evaluation metrics
        result = evaluate_model(model, X_test_samples, y_test_samples)
        print("\nPerformance Metrics:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Average Precision: {result['avg_precision']:.4f}")
        print(f"Average Recall: {result['avg_recall']:.4f}")
        print(f"Average F1 Score: {result['avg_f1_score']:.4f}")

if __name__ == "__main__":
    # Load preprocessed data
    X = np.load('processed_features.npy').astype(np.float32)
    y = np.load('processed_labels.npy')
    
    # Split dataset (using the same logic as before)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2)

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']

    test_models(X_test, y_test, genres)
