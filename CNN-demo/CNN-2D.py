#coding:utf-8

'''
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py
'''

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

import random, cPickle


from nlp_code.get_data import input_data


# load data
data, label = input_data()
# shuffle data
index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
data = data/255
label = label[index]
print(data.shape[0], ' samples')

# 20 classes, transform labels into the format requirement of keras is binary class matrices
label = np_utils.to_categorical(label, 20)
# print (data)
###############
# start to build CNN model
###############

max_features = 10000
embedding_dims = 100
maxlen = 100


#生成一个model
model = Sequential()


# First conv layer, 4 conv-cores with 5x5 and 1 chanel

# model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
# model.add(Dropout(0.5))

# model.add(Convolution2D(4, 5, 5, border_mode='valid', input_shape=data.shape[-3:], activation="relu"))
model.add(Convolution2D(4, 5, 5, border_mode='valid', input_shape=data.shape[-3:], activation="relu"))
# model.add(Activation('sigmoid'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Second conv layer, 8 conv-cores with 3x3 and 4 chanel
model.add(Convolution2D(8, 3, 3, border_mode='valid', activation="relu"))
model.add(Activation('relu'))
# model.add(Activation('sigmoid'))
model.add(Dropout(0.5))

model.add(MaxPooling2D(pool_size=(2, 2)))

# First conv layer, 16 conv-cores with 5x5 and 1 chanel

model.add(Convolution2D(16, 3, 3, border_mode='valid', activation="relu"))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Using fc layer to flatten the feature maps
model.add(Flatten())
model.add(Dense(128, init='normal'))
model.add(Activation('tanh'))
# model.add(Activation('sigmoid'))

# Using softmax to do classification
model.add(Dense(20, init='normal'))
model.add(Activation('softmax'))


#############
# train model
##############
# using SGD + momentum
sgd = SGD(l2=0.0, lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode="categorical")
# model.compile(loss='mean_squared_error', optimizer='sgd', class_mode="categorical")


hist = model.fit(data, label, batch_size=128, nb_epoch=100, shuffle=True, verbose=1, show_accuracy=True, validation_split=0.0)
