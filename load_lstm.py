import cPickle
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
__author__ = 'bohaohan'
from get_data import input_data
max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
X_train, y_train, X_test, y_test = input_data()
print y_test
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
model = Sequential()
# model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.5))
model.add(Embedding(max_features, 128, input_length=maxlen))
# model.add(LSTM(128, dropout_W=0.5, dropout_U=0.5))  # try using a GRU instead, for fun
model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               class_mode="binary")
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model.load_weights('lstm_w.h5')
# print model.evaluate(X_test, y_test, batch_size=32, show_accuracy=True)
# print model.predict_proba(X_test)
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')