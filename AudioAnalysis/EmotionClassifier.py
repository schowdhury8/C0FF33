import os
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from SpeechFeatureData import SpeechFeatureData 

# Turn off TF verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

speech_features = SpeechFeatureData()
speech_features.load_preprocess_data()

# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9, epsilon=1e-8, decay=0.
# SGD    : lr=0.01, momentum=0., decay=0.
opt = Adam()

batch_size = 35
nb_epochs = 400

print("Training X shape: " + str(speech_features.train_X.shape))
print("Training Y shape: " + str(speech_features.train_Y.shape))
print("Test X shape: " + str(speech_features.test_X.shape))
print("Test Y shape: " + str(speech_features.test_X.shape))

input_shape = (speech_features.train_X.shape[1], speech_features.train_X.shape[2])
print('Building LSTM RNN model ...')
model = Sequential()
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=speech_features.train_Y.shape[1], activation='softmax'))

if os.path.isfile('./weights/model_weights.h5'):
    model.load_weights('./weights/model_weights.h5')

print("Compiling ...")
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

print("Training ...")
model.fit(speech_features.train_X, speech_features.train_Y, batch_size=batch_size, epochs=nb_epochs)

print("\nTesting ...")
score, accuracy = model.evaluate(speech_features.test_X, speech_features.test_Y, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

