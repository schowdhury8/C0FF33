import numpy as np
import librosa
import math
import re
import os

class SpeechFeatureData:

    'Speech audio features for emotion classification'
    hop_length = None
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

    dir_trainfolder = './speech_data/training'
    dir_testfolder = './speech_data/testing'
    dir_all_files = './speech_data'

    train_X_preprocessed_data = 'data_train_input.npy'
    train_Y_preprocessed_data = 'data_train_target.npy'
    test_X_preprocessed_data = 'data_test_input.npy'
    test_Y_preprocessed_data = 'data_test_target.npy'

    train_X = train_Y = None
    test_X = test_Y = None

    def __init__(self):
        self.hop_length = 512

    def load_preprocess_data(self):
        self.trainfiles_list = self.path_to_audiofiles(self.dir_trainfolder)
        self.testfiles_list = self.path_to_audiofiles(self.dir_testfolder)

        all_files_list = []
        all_files_list.extend(self.trainfiles_list)
        all_files_list.extend(self.testfiles_list)


        print('Total number of speech data files: ' + str(len(all_files_list)))

        if os.path.isfile(self.train_X_preprocessed_data):
            print('Training features found, continuing...')
            self.train_X = np.load(self.train_X_preprocessed_data)
            self.train_Y = np.load(self.train_Y_preprocessed_data)
        else:
            # Training set
            print('Extracting training features: ')
            self.train_X, self.train_Y = self.extract_audio_features(self.trainfiles_list)
            with open(self.train_X_preprocessed_data, 'wb') as f:
                np.save(f, self.train_X)
            with open(self.train_Y_preprocessed_data, 'wb') as f:
                self.train_Y = self.one_hot(self.train_Y)
                np.save(f, self.train_Y)

        if os.path.isfile(self.test_X_preprocessed_data):
            print('Testing features found, continuing...')
            self.test_X = np.load(self.test_X_preprocessed_data)
            self.test_Y = np.load(self.test_Y_preprocessed_data)
        else:
            # Testing set
            print('Extracting testing features: ')
            self.test_X, self.test_Y = self.extract_audio_features(self.testfiles_list)
            with open(self.test_X_preprocessed_data, 'wb') as f:
                np.save(f, self.test_X)
            with open(self.test_Y_preprocessed_data, 'wb') as f:
                self.test_Y = self.one_hot(self.test_Y)
                np.save(f, self.test_Y)

    def load_deserialize_data(self):

        self.train_X = np.load(self.train_X_preprocessed_data)
        self.train_Y = np.load(self.train_Y_preprocessed_data)

        self.test_X = np.load(self.test_X_preprocessed_data)
        self.test_Y = np.load(self.test_Y_preprocessed_data)

    def extract_audio_features(self, list_of_audiofiles):
        timeseries_length = 128
        data = np.zeros((len(list_of_audiofiles), timeseries_length, 33), dtype=np.float64)
        target = []

        for i, file in enumerate(list_of_audiofiles):
            y, sr = librosa.load(file)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=self.hop_length, n_mfcc=13)
            spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)

            splits = re.split('[.]', file)
            features = re.split('/', splits[1])[4]
            emotion = self.feature_conversion(features)
            #print(emotion, features)
            target.append(emotion)
          
            if mfcc.T[0:128, :].shape != (128, 13): 
                continue
            
            data[i, :, 0:13] = mfcc.T[0:timeseries_length, :]
            data[i, :, 13:14] = spectral_center.T[0:timeseries_length, :]
            data[i, :, 14:26] = chroma.T[0:timeseries_length, :]
            data[i, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]

            print("Extracting features from speech file %i of %i." % (i + 1, len(list_of_audiofiles)))

        return data, np.expand_dims(np.asarray(target), axis=1)

    def one_hot(self, Y_emotion_strings):
        y_one_hot = np.zeros((Y_emotion_strings.shape[0], len(self.emotions)))
        for i, emotion_string in enumerate(Y_emotion_strings):
            index = self.emotions.index(emotion_string)
            y_one_hot[i, index] = 1
        return y_one_hot

    def path_to_audiofiles(self, dir_folder):
        list_of_audio = []
        for actor in os.listdir(dir_folder):
            actor_folder = dir_folder+'/'+actor
            for file in os.listdir(actor_folder):
                if file.endswith(".wav"):
                    directory = actor_folder+'/'+file
                    list_of_audio.append(directory)
        return list_of_audio

    def feature_conversion(self, features):
        # Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
        # Vocal channel (01 = speech, 02 = song).
        # Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
        # Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
        # Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
        # Repetition (01 = 1st repetition, 02 = 2nd repetition).
        # Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

        emotions = {
            1: 'neutral',
            2: 'calm', 
            3: 'happy', 
            4: 'sad', 
            5: 'angry', 
            6: 'fearful', 
            7: 'disgust', 
            8: 'surprised'
            }

        features = re.split('-', features)
        emotion = emotions[int(features[2])]
        return emotion