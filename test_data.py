# coding: utf-8
import re
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

__author__ = 'bohaohan'
# from keras.datasets import imdb
# from nltk.stem import WordNetLemmatizer
# (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=20,
#                                                       test_split=0.2)
# for i in X_test:
#     print i

# print WordNetLemmatizer().lemmatize("lives")
# import nltk
# nltk.download()
from keras.datasets import imdb, reuters
from get_data import input_data
max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
# (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
# (X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=1000, test_split=0.2)
# X_train, y_train, X_test, y_test = input_data()
# print(len(X_train), 'train sey_trainquences')
# print(len(X_test), 'test sequences')
# print(X_train[0], 'train sequences')
# tokenizer = Tokenizer(nb_words=1000)
# X_train = sequence.pad_sequences(X_train, maxlen=100)
# print(X_train[0], 'train sequences')
from keras.preprocessing.text import one_hot
x = "你 我 他"
print one_hot(n=10000, text=x)