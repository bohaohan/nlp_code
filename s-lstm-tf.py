from get_data import input_data_gen_w2v

__author__ = 'bohaohan'
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

# print('Loading data...')
# x_train, y_train, x_val, y_val = input_data_gen_w2v()
# print "end load"

data_dim = 300
# timesteps = len(x_train[0])
timesteps = 100
nb_classes = 3
batch_size = 16
print "build model"
# expected input batch shape: (batch_size, timesteps, data_dim)
# note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print "finish model"
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, nb_classes))

# generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, nb_classes))


model.fit(x_train, y_train,
          batch_size=batch_size, nb_epoch=2000, show_accuracy=True,
          validation_data=(x_val, y_val))