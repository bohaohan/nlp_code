'''Train a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF+LogReg.
Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
GPU command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py
'''

from __future__ import print_function
import cPickle
import numpy as np
np.random.seed(1337)  # for reproducibility
import sys
sys.setrecursionlimit(1000000)
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb
from nlp_code.get_data import input_data
max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
# (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
#                                                       test_split=0.2)
X_train, y_train, X_test, y_test = input_data()
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
#(samples, dim)
print('X_test shape:', X_test.shape)
#Y  1, 0
print('Build model...')
model = Sequential()
# model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.5))
model.add(Embedding(max_features, 128, input_length=maxlen))
# model.add(LSTM(128, dropout_W=0.5, dropout_U=0.5))  # try using a GRU instead, for fun
model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=150,
          validation_data=(X_test, y_test), show_accuracy=True)
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size,
                            show_accuracy=True)
print(y_test)
print(model.predict_classes(X_test))

# model.evaluate()
print('Test score:', score)
print('Test accuracy:', acc)
model.save_weights('lstm_w1.h5')
