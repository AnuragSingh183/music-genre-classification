from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import joblib

# Load saved data
X = np.load('processed_features.npy').astype(np.float32)
y = np.load('processed_labels.npy')
filenames = np.load('file_names.npy')

# Sort data for consistency
sorted_indices = np.argsort(filenames)
X, y, filenames = X[sorted_indices], y[sorted_indices], filenames[sorted_indices]

# Load models
MODEL_PATH_GRU = 'saved_models/gru_model.joblib'
MODEL_PATH_VANILLA = 'saved_models/vanilla_model.joblib'

gru_model = joblib.load(MODEL_PATH_GRU)
vanilla_model = joblib.load(MODEL_PATH_VANILLA)

# Genre mapping
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def predict_with_model(model, file_name):
    """Common prediction logic for both models."""
    if file_name not in filenames:
        return jsonify({'error': f'File {file_name} not found in dataset'}), 404

    # Retrieve corresponding feature vector
    file_index = np.where(filenames == file_name)[0][0]
    features = np.expand_dims(X[file_index], axis=0)  # Add batch dimension

    # Predict genre and confidence
    predicted_index, confidence = model.predict_with_confidence(features)
    predicted_genre = GENRES[int(predicted_index[0])]

    return {
        'predicted_genre': predicted_genre,
        'confidence': round(float(confidence[0]) * 100, 2)
    }

@app.route('/predict_gru', methods=['POST'])
def predict_gru():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    return jsonify(predict_with_model(gru_model, audio_file.filename))

@app.route('/predict_vanilla', methods=['POST'])
def predict_vanilla():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    return jsonify(predict_with_model(vanilla_model, audio_file.filename))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
