"An example of predicting a music emotion from a custom audio file"
import sys
import imp
import librosa
import numpy as np
import keras

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

model_dir = './model.h5'

def extract_audio_features(file):
    "Extract audio features from an audio file"
    timeseries_length = 128
    features = np.zeros((1, timeseries_length, 33), dtype=np.float64)

    y, sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)

    features[0, :, 0:13] = mfcc.T[0:timeseries_length, :]
    features[0, :, 13:14] = spectral_center.T[0:timeseries_length, :]
    features[0, :, 14:26] = chroma.T[0:timeseries_length, :]
    features[0, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]
    return features

def get_emotion(model, music_path):
    "Predict emotion of music using a trained model"
    predictions = model.predict(extract_audio_features(music_path))
    predict_emotion = EMOTIONS[np.argmax(predictions)]
    return predictions, predict_emotion

if __name__ == '__main__':
    PATH = sys.argv[1] #if len(sys.argv) == 2 else './audios/classical_music.mp3'
    MODEL = keras.models.load_model(model_dir)
    PREDICTIONS, EMOTION = get_emotion(MODEL, PATH)
    print(str(EMOTIONS)[1:-1].replace(',','').replace("'", ''))
    print(PREDICTIONS)
    print('Model prediction: ' + EMOTION)