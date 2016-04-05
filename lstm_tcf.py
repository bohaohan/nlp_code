from keras.preprocessing import sequence
from keras.utils import np_utils

from get_data import input_data_gen_w2v, input_data_w2v, input_data, input_data_2

__author__ = 'bohaohan'
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
import numpy as np
print('Loading data...')
# maxlen = 200
x_train, y_train, x_val, y_val = input_data_2()
maxlen =  max(len(x) for x in x_train)
maxlen2 =  max(len(x) for x in x_val)
if maxlen < maxlen2:
    maxlen = maxlen2
print "end load"
X_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', x_val.shape)

print('y_train shape:', y_train.shape)
print('y_val shape:', y_val.shape)
data_dim = 300
timesteps = len(x_train[0])
nb_classes = 2
nb_epoch = 2000
print "build model"
# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()

model.add(Embedding(10000, 128, input_length=timesteps))

model.add(LSTM(200, return_sequences=True, activation='sigmoid', inner_activation='hard_sigmoid'))
               # ,input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(100, return_sequences=True, activation='sigmoid', inner_activation='hard_sigmoid'))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#binary_crossentropy
#categorical_crossentropy
print "finish model"
# generate dummy training data


model.fit(x_train, y_train,
          batch_size=64, nb_epoch=nb_epoch, show_accuracy=True,
          validation_data=(x_val, y_val))