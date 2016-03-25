# coding: utf-8
import random
from keras.preprocessing.text import one_hot

__author__ = 'bohaohan'
import jieba


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
            try:
                tag = tks[1]
                if tag == "预警\n":
                    tag = 1
                else:
                    tag = 0
                train_words.append(x)
                train_tags.append(tag)
            except:
                pass
    # print train_words[0]
    index = [i for i in range(len(train_words))]
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


if __name__ == "__main__":
    X, Y, test_words, test_tags = input_data()
    print Y