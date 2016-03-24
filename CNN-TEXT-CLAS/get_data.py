# coding: utf-8
import random
from keras.preprocessing.text import one_hot

__author__ = 'bohaohan'
import jieba
import numpy as np

def pad_sentences(sentences, padding_word=0):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def input_data():
    train_file = "3.17data.txt"
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


if __name__ == "__main__":
    X, Y, test_words, test_tags = input_data()
    print X[0]