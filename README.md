This is a Flask-based web application that predicts the genre of uploaded .wav audio files. The application supports two models:
GRU Model
Vanilla RNN Model
The app features a sleek, dark-themed UI with options to select your desired model for prediction.

Features-
Predicts music genres from .wav audio files.
Supports both GRU and Vanilla RNN models.
Displays prediction confidence percentage for accuracy insights.
Clean, user-friendly web interface for seamless experience.

Installation and Setup-
git clone https://github.com/your-username/music-genre-classifier.git
cd music-genre-classifier

To Train the model:
    Run main.py file with the model name as parameter.

    Example Usage:
        python main.py vanilla
        python main.py gru

To Test the models:
    After training you can test the model by running the test_models.py file.

    Example usage:
        python test_models.py
    
    This will give the evaluation metrics and plot of confusion matrix for both the models.

To run the App/Frontend:

    python app.py

    Upload any file from the genres dataset. Should be .wav file.

![image](https://github.com/user-attachments/assets/6afa67ee-8776-45df-a8fc-a64cc79f902b)

Following libraries have been used:
matplotlib
flask
joblib
SciPy
librosa

Note: We have included the feature extraction and saved model files in the source code, you can use them or train the model again.

Dataset Link: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Note: We have just used the .wav files from the dataset which are in the genres_original folder in the above link.
