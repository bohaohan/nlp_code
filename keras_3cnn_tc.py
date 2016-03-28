from keras.models import Sequential
from keras.layers import Merge, LSTM, Dense, Convolution2D, MaxPooling2D, Activation, Dropout, Flatten
from keras.utils import np_utils
import numpy as np
from get_data import input_data_gen_w2v2


nb_classes = 3
# x_train, y_train, x_val, y_val = input_data_gen_w2v2()
x_train, y_train, x_val, y_val = [],[],[],[]
y_train = np_utils.to_categorical(y_train, 3)
y_val = np_utils.to_categorical(y_val, 3)
# se_len = len(x_train[0])
se_len = 55
# print x_train.shape

encoder_a = Sequential()
encoder_a.add(Convolution2D(nb_filter=300, nb_row=3, nb_col=300, border_mode='valid', input_shape=(1, se_len, 300), activation="relu"))
encoder_a.add(Activation('relu'))
encoder_a.add(MaxPooling2D(pool_size=(se_len-2, 1), strides=(1, 1)))

encoder_b = Sequential()
encoder_b.add(Convolution2D(nb_filter=300, nb_row=4, nb_col=300, border_mode='valid', input_shape=(1, se_len, 300), activation="relu"))
encoder_b.add(Activation('relu'))
encoder_b.add(MaxPooling2D(pool_size=(se_len-3, 1), strides=(1, 1)))

encoder_c = Sequential()
encoder_c.add(Convolution2D(nb_filter=300, nb_row=5, nb_col=300, border_mode='valid', input_shape=(1, se_len, 300), activation="relu"))
encoder_c.add(Activation('relu'))
encoder_c.add(MaxPooling2D(pool_size=(se_len-4, 1), strides=(1, 1)))


decoder = Sequential()
decoder.add(Merge([encoder_a, encoder_b, encoder_c], mode='concat'))
decoder.add(Flatten())
decoder.add(Dropout(0.5))
decoder.add(Dense(32, activation='relu'))
decoder.add(Dense(nb_classes, activation='softmax'))

decoder.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# # generate dummy training data
# x_train_a = np.random.random((1000, timesteps, data_dim))
# x_train_b = np.random.random((1000, timesteps, data_dim))
# y_train = np.random.random((1000, nb_classes))
#
# # generate dummy validation data
# x_val_a = np.random.random((100, timesteps, data_dim))
# x_val_b = np.random.random((100, timesteps, data_dim))
# y_val = np.random.random((100, nb_classes))

decoder.fit([x_train, x_train, x_train], y_train,
            batch_size=64, nb_epoch=30, show_accuracy=True,
            validation_data=([x_val, x_val, x_val], y_val))