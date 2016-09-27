# coding: utf-8
import random
from keras.preprocessing.text import one_hot

__author__ = 'bohaohan'
# import jieba
import numpy as np


def pad_sentences(sentences, padding_word=0):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    print len(sentences)
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def input_data_gen():
    train_file = "total-data.txt"
    train_words = []
    train_tags = []

    X = []
    Y = []

    test_words = []
    test_tags = []
    with open(train_file, 'r') as f1:
        for line in f1:

            tks = line.split('-0-')
            word = tks[0]
            x = one_hot(n=10000, text=word)
            tag = tks[1]

            if tag == "+":
                tag = [1, 0, 0]
            elif tag == "-":
                tag = [0, 1, 0]
            else:
                tag = [0, 0, 1]

            train_words.append(x)
            train_tags.append(tag)

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

    return X, Y, test_words, test_tags


def input_data():
    train_file = "3.25-data.txt"

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
            words = ""

            for i in word:
                words += i + " "
            words = words[:len(words)-1].encode('utf8')
            x = one_hot(n=10000, text=words)

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

            except Exception as e:
                print e.message

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

    return X, Y, test_words, test_tags


def get_label_rm(label):

    if label == "+":
        label = [1, 0]
    else:
        label = [0, 1]
    return label


def get_label_qa(label):
    tag = [0 for i in range(6)]
    tag[int(label)] = 1
    return tag


def get_input_data(train_file="rm_result.txt", test_file=None, split=0.1, label_func=get_label_rm):

    X = []
    Y = []
    train_words = []
    train_tags = []
    test_len = 0
    if test_file is not None:
        with open(test_file, 'r') as f1:

            for line in f1:
                line = line.replace("\n", "")
                tks = line.split('-0-')
                word = tks[0]
                x = one_hot(n=10000, text=word)

                if len(x) > 500:
                    continue
                try:
                    tag = label_func(tks[1])

                    train_words.append(x)
                    train_tags.append(tag)
                except Exception as e:
                    print e.message
        test_len = len(train_words)

    with open(train_file, 'r') as f1:

        for line in f1:
            line = line.replace("\n", "")
            tks = line.split('-0-')
            word = tks[0]

            x = one_hot(n=10000, text=word)

            if len(x) > 500:
                continue
            try:
                tag = label_func(tks[1])

                train_words.append(x)
                train_tags.append(tag)
            except Exception as e:
                    print e.message

    index = [i for i in range(len(train_words))]
    print "padding"
    train_words = pad_sentences(train_words)
    train_tags = np.concatenate([train_tags], 0)
    print "end padding"

    if test_file is None:
        random.shuffle(index)
        test_len = int(split * len(train_words))

    test_words = []
    test_tags = []

    for i, j in enumerate(train_words):
        if i < test_len:
            test_words.append(train_words[index[i]])
            test_tags.append(train_tags[index[i]])
        else:
            X.append(train_words[index[i]])
            Y.append(train_tags[index[i]])

    return X, Y, test_words, test_tags


if __name__ == "__main__":
    x_train, y_train, x_dev, y_dev = get_input_data()
    print y_dev