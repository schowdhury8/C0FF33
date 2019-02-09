import os
from keras.models import load_model
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

from SpeechFeatureData import SpeechFeatureData 

model_dir = './model.h5'

# Turn off TF verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

speech_features = SpeechFeatureData()
speech_features.load_preprocess_data()

batch_size = 35
epochs = 400

print('Training X shape: ' + str(speech_features.train_X.shape))
print('Training Y shape: ' + str(speech_features.train_Y.shape))
print('Test X shape:     ' + str(speech_features.test_X.shape))
print('Test Y shape:     ' + str(speech_features.test_X.shape))

input_shape = (speech_features.train_X.shape[1], speech_features.train_X.shape[2])

if not os.path.isfile('./model.h5'):
    print('Building LSTM RNN model ...')
    model = Sequential()
    model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(Dense(units=speech_features.train_Y.shape[1], activation='softmax'))

    print('Compiling...')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
else:
    print('Found saved model..')
    model = load_model(model_dir)
 

checkpoint = ModelCheckpoint(model_dir, monitor='acc', verbose=1, save_weights_only=False, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print('Training...')
model.fit(speech_features.train_X, speech_features.train_Y, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list)

print('\nTesting...')
score, accuracy = model.evaluate(speech_features.test_X, speech_features.test_Y, batch_size=batch_size, verbose=1)
print('Test loss:      ', score)
print('Test accuracy:  ', accuracy)