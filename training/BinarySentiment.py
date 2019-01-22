from time import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard

import re

print('Reading in raw data...')
names = ["sentiment","content"]
dataframe = pd.read_csv('../datasets/posneg_processed.csv', names=names, encoding = "ISO-8859-1")
dataframe['content']=dataframe['content'].astype(str)
# 0 - neg, 1 - 
print('Tokenizing...')
max_features = 1000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(dataframe['content'].values)
X = tokenizer.texts_to_sequences(dataframe['content'].values)
X = pad_sequences(X)

Y = pd.get_dummies(dataframe['sentiment'])
print(Y)
#print(Y.idxmax(axis=1))

x, x_test, y, y_test = train_test_split(X,Y,test_size=0.2,train_size=0.8)
x_train, x_cv, y_train, y_cv = train_test_split(X,Y,test_size = 0.2,train_size =0.8,random_state=42)

#print(x.shape,y.shape)
embed_dim = 128
lstm_out = 128

model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.1))
model.add(LSTM(lstm_out, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

batch_size = 128
model.fit(x_train, y_train,batch_size=batch_size,epochs=50,validation_data=(x_cv, y_cv))

model.save('../models/binary_model.h5')  # creates a HDF5 file 'my_model.h5'

score,acc = model.evaluate(x_test, y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

del model  # deletes the existing model
