# coding: utf-8
import random
import gensim
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot

__author__ = 'bohaohan'
import jieba
import numpy as np
import sys
import nltk
from os import path
sys.path.append("..")
PATH = path.dirname(path.abspath(__file__))
PATH = PATH[:PATH.rfind('/')]
_W2V_BINARY_PATH = PATH + "/word2vec/GoogleNews-vectors-negative300.bin"


def pad_sentences(sentences, padding_word=0):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    # sequence_length = 100
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [np.zeros([300, ], dtype=np.float32).reshape(300, 1)] * num_padding

        padded_sentences.append(new_sentence)
    return padded_sentences


def input_data():
    train_file = "3.25-data.txt"
    test_file = "test.txt"

    train_words = []
    train_tags = []

    X = []
    Y = []

    test_words = []
    test_tags = []
    with open(train_file, 'r') as f1:
        for line in f1:
            tks = line.split('\t', 1)
            word = tks[0]
            word = jieba.cut(word, cut_all=True)
            words = ""
            for i in word:
                words += i + " "
            words = words[:len(words)-1].encode('utf8')
            x = one_hot(n=10000, text=words)
            if len(x) > 300:
                print len(x)
            try:
                tag = tks[1]
                if tag == "预警\n":
                    tag = [1, 0]
                else:
                    tag = [0, 1]
                train_words.append(x)
                train_tags.append(tag)
            except:
                pass
    # print train_words[0]
    index = [i for i in range(len(train_words))]
    train_words = pad_sentences(train_words)
    train_tags = np.concatenate([train_tags], 0)
    random.shuffle(index)
    for i, j in enumerate(train_words):
        if i < 0.1 * len(train_words):
            test_words.append(train_words[index[i]])
            test_tags.append(train_tags[index[i]])
        else:
            X.append(train_words[index[i]])
            Y.append(train_tags[index[i]])

    # with open(test_file, 'r') as f1:
    #     for line in f1:
    #         tks = line.split('\t', 1)
    #         word = tks[0]
    #         tag = tks[1]
    #         test_words.append(word)
    #         test_tags.append(tag)
    return X, Y, test_words, test_tags


def input_data_w2v(train_file="3.25-data.txt", split=0.1):

    model = get_word2vec()
    train_words = []
    train_tags = []

    with open(train_file, 'r') as f1:
        for line in f1:
            tks = line.split('\t', 1)
            word = tks[0]
            words = jieba.cut(word, cut_all=True)

            x = []
            # words = ""
            for word in words:
                if word in model:
                    x.append(model[word].reshape(300, 1))
                else:
                    x.append(np.zeros([300, ], dtype=np.float32).reshape(300, 1))

            if len(x) > 500:
                continue
            try:
                tag = tks[1]
                if tag == "预警\n":
                    tag = [1, 0]
                else:
                    tag = [0, 1]
                train_words.append(x)
                train_tags.append(tag)
            except:
                pass
    # print train_words[0]
    index = [i for i in range(len(train_words))]
    print "padding"
    train_words = pad_sentences(train_words)
    train_tags = np.concatenate([train_tags], 0)
    print "end padding"
    random.shuffle(index)
    test_len = int(split * len(train_words))
    train_len = len(train_words) - test_len
    test_words = np.zeros([test_len + 1, len(train_words[0]), 300, 1], dtype=np.float32)
    test_tags = np.zeros([test_len + 1, 2], dtype=np.float32)
    X = np.zeros([train_len, len(train_words[0]), 300, 1], dtype=np.float32)
    Y = np.zeros([train_len, 2], dtype=np.float32)
    for i, j in enumerate(train_words):
        if i < test_len:
            test_words[i] = train_words[index[i]]
            test_tags[i] = train_tags[index[i]]
        else:
            X[i - test_len] = train_words[index[i]]
            Y[i - test_len] = train_tags[index[i]]

    return X, Y, test_words, test_tags


def input_data_gen_w2v(train_file="gene-data.txt", split=0.1):

    model = get_word2vec()
    train_words = []
    train_tags = []

    with open(train_file, 'r') as f1:
        for line in f1:
            tks = line.split('-0-')
            word = tks[0]
            words = nltk.word_tokenize(word)

            x = []
            # words = ""
            for word in words:
                if word in model:
                    x.append(model[word].reshape(300, 1))
                else:
                    x.append(np.zeros([300, ], dtype=np.float32).reshape(300, 1))

            if len(x) > 500:
                continue
            try:
                tag = tks[1]
                if tag == "+":
                    tag = [1, 0, 0]
                elif tag == "-":
                    tag = [0, 1, 0]
                else:
                    tag = [0, 0, 1]
                train_words.append(x)
                train_tags.append(tag)
            except:
                pass
    # print train_words[0]
    index = [i for i in range(len(train_words))]
    print "padding"
    train_words = pad_sentences(train_words)
    train_tags = np.concatenate([train_tags], 0)
    print "end padding"
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

    return X, Y, test_words, test_tags

def get_word2vec():
    print "load model"
    model = gensim.models.Word2Vec.load_word2vec_format(_W2V_BINARY_PATH, binary=True)
    # positive = model['positive']
    # print positive
    # print len(positive)
    print "finish load"
    return model


if __name__ == "__main__":
    X, Y, test_words, test_tags = input_data_gen_w2v()
    # X = np.array(X, dtype=np.float32)
    print X[0]
