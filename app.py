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

MODEL_PATH = 'saved_models/gru_model.joblib'

# Load the model
model = joblib.load(MODEL_PATH)

# Genre mapping
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

app = Flask(__name__)\

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_genre():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Find the corresponding sample
    file_name = audio_file.filename
    if file_name not in filenames:
        return jsonify({'error': f'File {file_name} not found in dataset'}), 404

    # Retrieve corresponding feature vector
    file_index = np.where(filenames == file_name)[0][0]
    features = np.expand_dims(X[file_index], axis=0)  # Add batch dimension

    # Predict genre and confidence
    predicted_index, confidence = model.predict_with_confidence(features)
    predicted_genre = GENRES[int(predicted_index[0])]

    return jsonify({
        'predicted_genre': predicted_genre,
        'confidence': round(float(confidence[0]) * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
