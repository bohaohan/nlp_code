import random
import numpy as np
__author__ = 'bohaohan'
import cPickle
# from get_data_wordv import input_data_gen_w2v
import csv

file_name = "data.pkl"


# def write_file():
#     x_train, y_train, x_dev, y_dev = input_data_gen_w2v(split=0)
#
#     print len(x_train), len(y_train)
#     data = []
#     data.append(x_train)
#     data.append(y_train)
#     cPickle.dump(data, open("./pkl/data.pkl", "wb"))


def load_data(split=0.1):
    print "loading data"
    data = cPickle.load(open("./pkl/data.pkl", "rb"))
    train_words = data[0]
    train_tags = data[1]
    print "end loading data"
    print "seperating data"
    index = [i for i in range(len(train_words))]
    random.shuffle(index)
    test_len = int(split * len(train_words))
    train_len = len(train_words) - test_len
    test_words = np.zeros([test_len + 1, len(train_words[0]), 300, 1], dtype=np.float32)
    test_tags = np.zeros([test_len + 1, 3], dtype=np.float32)
    X = np.zeros([train_len, len(train_words[0]), 300, 1], dtype=np.float32)
    Y = np.zeros([train_len, 3], dtype=np.float32)
    for i, j in enumerate(train_words):
        if i < test_len:
            test_words[i] = train_words[index[i]]
            test_tags[i] = train_tags[index[i]]
        else:
            X[i - test_len] = train_words[index[i]]
            Y[i - test_len] = train_tags[index[i]]
    print "end seperating data"
    return X, Y, test_words, test_tags

if __name__ == "__main__":
    X, Y, test_words, test_tags = load_data()
    print len(X)