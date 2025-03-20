import numpy as np
import os
import sys
import joblib  # Efficient model saving
from data_processing import load_dataset
from new_evaluate import evaluate_model, plot_confusion_matrix, plot_metrics, train_test_split
from rnn_cells import RNNModel

# File paths for saved features
FEATURES_FILE = 'processed_features.npy'
LABELS_FILE = 'processed_labels.npy'

# Directory to save models
MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)

import joblib
import os

def save_model(model, model_name):
    """Save model with optimized size using float32 and joblib compression."""
    # Convert model parameters to float32 before saving
    for param_name, param in vars(model).items():
        if isinstance(param, np.ndarray):
            setattr(model, param_name, param.astype(np.float32))
    
    # Clear all caches
    for attr in ['h_cache', 'x_cache', 'softmax_cache', 'y_cache']:
        if hasattr(model, attr):
            setattr(model, attr, [])
    
    # Clear caches in RNN cells
    if hasattr(model, 'rnn_cell'):
        if hasattr(model.rnn_cell, 'reset_memory'):
            model.rnn_cell.reset_memory()
        if hasattr(model.rnn_cell, 'cache'):
            model.rnn_cell.cache = []
        if hasattr(model.rnn_cell, 'h_cache'):
            model.rnn_cell.h_cache = []
        if hasattr(model.rnn_cell, 'x_cache'):
            model.rnn_cell.x_cache = []

    model_file = os.path.join(MODEL_DIR, f"{model_name}_model.joblib")
    joblib.dump(model, model_file, compress=3)
    print(f"Model saved as {model_file}")

def train_model(model_type):
    # Set parameters
    data_dir = 'genres'  # Update with your path
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Load preprocessed dataset
    print("Loading and preprocessing dataset...")
    if not (os.path.exists(FEATURES_FILE) and os.path.exists(LABELS_FILE)):
        X, y = load_dataset(data_dir, genres)
        np.save(FEATURES_FILE, X.astype(np.float32))  # Convert to float32 for memory efficiency
        np.save(LABELS_FILE, y)
    else:
        print("Loading saved features...")
        X = np.load(FEATURES_FILE).astype(np.float32)
        y = np.load(LABELS_FILE)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Model parameters
    input_size = X_train.shape[2]  # Number of features
    hidden_size = 128
    output_size = len(genres)
    
    # Train model
    print(f"\nTraining {model_type.upper()} model...")
    model = RNNModel(input_size, hidden_size, output_size, cell_type=model_type)

    # Clear caches before training starts (important for reducing model size)
    if hasattr(model.rnn_cell, 'reset_memory'):
        model.rnn_cell.reset_memory()
    model.y_cache = []
    model.softmax_cache = []

    # Train model
    history = model.train(X_train, y_train, epochs=60, batch_size=32, learn_rate=0.001)

    # Clear caches after training (important for reducing model size)
    if hasattr(model.rnn_cell, 'reset_memory'):
        model.rnn_cell.reset_memory()
    model.y_cache = []
    model.softmax_cache = []

    # Save model
    save_model(model, model_type)
    
    # Evaluate model
    print(f"\nEvaluating {model_type.upper()} model...")
    result = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Average Precision: {result['avg_precision']:.4f}")
    print(f"Average Recall: {result['avg_recall']:.4f}")
    print(f"Average F1 Score: {result['avg_f1_score']:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(result['confusion_matrix'], genres, 
                         title=f'Confusion Matrix - {model_type.upper()}')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <model_type>")
        print("Options: vanilla | gru | birnn")
        sys.exit(1)
    
    model_type = sys.argv[1].lower()
    if model_type not in ['vanilla', 'gru', 'birnn']:
        print("Invalid model type. Choose from: vanilla | gru | birnn")
        sys.exit(1)
    
    train_model(model_type)
