from keras.models import Sequential
from keras.layers import Merge, Dense, Convolution2D, MaxPooling2D, Activation, Dropout, Flatten

from keras.utils import np_utils
from nlp_code.get_data import input_data_gen_w2v3



filter_sizes = [2, 3, 4, 5]
nb_classes = 2
batch_size = 64
np_epoch = 100
nb_filter = 100
word_vec_len = 300

x_train, y_train, x_val, y_val = input_data_gen_w2v3()

y_train = np_utils.to_categorical(y_train, nb_classes)
y_val = np_utils.to_categorical(y_val, nb_classes)
se_len = len(x_train[0][0])
# se_len = 55
# print x_train.shape
print x_train.shape, "x sh"
print x_val.shape, "xv sh"
print y_train.shape, "y sh"
print y_val.shape, "yv sh"
merge_array = []


for filter_size in filter_sizes:
    sub_conv = Sequential()
    sub_conv.add(Convolution2D(nb_filter=nb_filter, nb_row=filter_size, nb_col=word_vec_len, border_mode='valid', input_shape=(1, se_len, word_vec_len), activation="relu"))
    sub_conv.add(Activation('relu'))
    sub_conv.add(MaxPooling2D(pool_size=(se_len-filter_size+1, 1), strides=(1, 1)))
    merge_array.append(sub_conv)


decoder = Sequential()
decoder.add(Merge(merge_array, mode='concat'))
decoder.add(Flatten())
decoder.add(Dropout(0.5))
decoder.add(Dense(16, activation='relu'))
decoder.add(Dense(nb_classes, activation='softmax'))

decoder.compile(loss='categorical_crossentropy', optimizer='rmsprop')


decoder.fit([x_train, x_train, x_train, x_train], y_train,
            batch_size=batch_size, nb_epoch=np_epoch, show_accuracy=True,
            validation_data=([x_val, x_val, x_val, x_val], y_val))