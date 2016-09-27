from nlp_code.get_data import input_data_gen_w2v, input_data_w2v, input_data

__author__ = 'bohaohan'
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
import numpy as np
print('Loading data...')
x_train, y_train, x_val, y_val = input_data()
print "end load"

data_dim = 300
timesteps = len(x_train[0])
nb_classes = 1
nb_epoch = 2000
print "build model"
# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()

model.add(Embedding(1, 128, input_length=timesteps))

model.add(LSTM(200, return_sequences=True))
               # ,input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(100, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print "finish model"
# generate dummy training data


model.fit(x_train, y_train,
          batch_size=64, nb_epoch=nb_epoch, show_accuracy=True,
          validation_data=(x_val, y_val))