
# coding: utf-8

import sys
import cPickle
import jieba
import numpy
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import random
import nltk
import skflow

import sys
reload(sys)
sys.setdefaultencoding('utf8')


def input_data(train_file, test_file):
    train_words = []
    train_tags = []
    test_words = []
    test_tags = []
    with open(train_file, 'r') as f1:
        for line in f1:
            tks = line.split('\t', 1)
            word = tks[0]
            try:
                tag = tks[1]
                if tag == "预警\n":
                    tag = "预警"
                else:
                    tag = "垃圾"
                train_words.append(word)
                train_tags.append(tag)
            except:
                pass
    with open(test_file, 'r') as f1:
        for line in f1:
            tks = line.split('\t', 1)
            word = tks[0]
            tag = tks[1]
            test_words.append(word)
            test_tags.append(tag)
    return train_words, train_tags, test_words, test_tags


def input_data_gen(train_file, test_file):

    train_words = []
    train_tags = []
    test_words = []
    test_tags = []
    with open(train_file, 'r') as f1:
        for line in f1:
            tks = line.split('-0-')
            word = tks[0]
            try:
                tag = tks[1]
                if tag == "+":
                    tag = 0
                elif tag == "-":
                    tag = 1
                else:
                    tag = 2
                train_words.append(word)
                train_tags.append(tag)
            except:
                pass
    train_tags = np.concatenate([train_tags], 0)
    with open(test_file, 'r') as f1:
        for line in f1:
            tks = line.split('-0-', 1)
            word = tks[0]
            tag = tks[1]
            test_words.append(word)
            test_tags.append(tag)
    return train_words, train_tags, test_words, test_tags


def input_data_rm(train_file, test_file):

    train_words = []
    train_tags = []
    test_words = []
    test_tags = []
    with open(train_file, 'r') as f1:
        for line in f1:
            line = line.replace("\n", "")
            tks = line.split('-0-')
            word = tks[0]
            try:
                word = word.decode("utf-8")
                tag = tks[1]
                train_words.append(word)
                train_tags.append(tag)
            except:
                print word
                pass
    train_tags = np.concatenate([train_tags], 0)
    with open(test_file, 'r') as f1:
        for line in f1:
            line = line.replace("\n", "")
            tks = line.split('-0-', 1)
            word = tks[0]
            try:
                word = word.decode("utf-8")
                tag = tks[1]
                test_words.append(word)
                test_tags.append(tag)
            except:
                print word
    return train_words, train_tags, test_words, test_tags


comma_tokenizer = lambda x: nltk.word_tokenize(x)


def vectorize(train_words, test_words):
    v1 = HashingVectorizer(tokenizer=comma_tokenizer, n_features=10000, non_negative=True)
    v = make_pipeline(v1, TfidfTransformer())

    train_data = v.fit_transform(train_words)

    test_data = v.fit_transform(test_words)
    return train_data, test_data


def evaluate(actual, pred):
    m_precision = metrics.precision_score(actual, pred)
    m_recall = metrics.recall_score(actual, pred)
    print 'precision:{0:.3f}'.format(m_precision)
    print 'recall:{0:0.3f}'.format(m_recall)


def train_clf(train_data, train_tags):
    clf = MultinomialNB(alpha=0.01)
    clf.fit(train_data, numpy.asarray(train_tags))
    return clf

def train_svm(train_data, train_tags):
    # classifier = LinearSVC(C=0.5, penalty="l2", dual=False)
    # classifier = MultinomialNB(alpha=0.01)
    # classifier = LinearSVC()
    # random.seed(42)
    classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10],
        n_classes=6, batch_size=32, steps=9000, learning_rate=0.05)

    # classifier = SVC(kernel="linear")
    # classifier = MultinomialNB(alpha=0.000001)

    # classifier.fit(train_data, train_tags)
    classifier.fit(train_data.toarray(), train_tags)
    return classifier


def split_data(train_words, train_tags, k):
    index = [i for i in range(len(train_words))]
    random.shuffle(index)
    train_data = []
    train_tag = []
    test_data = []
    test_tag = []
    length = len(train_words)
    strip = int(0.1 * length)
    for i, j in enumerate(train_words):
        if (i >= k * strip) and (i < (k + 1) * strip):
            test_data.append(train_words[index[i]])
            test_tag.append(train_tags[index[i]])
        else:
            train_data.append(train_words[index[i]])
            train_tag.append(train_tags[index[i]])
    return train_data, train_tag, test_data, test_tag


def main():

    train_file = "data.txt"
    test_file = "test.txt"
    train_words, train_tags, test_words, test_tags = input_data_rm(train_file, test_file)
    train_data, test_data = vectorize(train_words, test_words)
    clf = train_svm(train_data, train_tags)
    # cPickle.dump(clf, open("svm.pkl", "wb"))
    pred = clf.predict(test_data)

    evaluate(numpy.asarray(test_tags), pred)

    k = 0
    for i, j in enumerate(pred):
        if pred[i].encode('utf8') == test_tags[i].encode('utf8'):
            k += 1
    print k

def test():
    train_file = "qa_train.txt"
    test_file = "qa_test.txt"

    train_word, train_tag, test_words, test_tags = input_data_rm(train_file, test_file)
    total_true = 0
    total_number = 0.0

    for i in range(10):
        print i, "test"

        train_words, train_tags, test_words, test_tags = train_word, train_tag, test_words, test_tags
        train_data, test_data = vectorize(train_words, test_words)
        train_tags = numpy.array(train_tags)
        test_tags = numpy.array(test_tags)
        clf = train_svm(train_data, train_tags)

        pred = clf.predict(test_data.toarray())

        k = 0
        al = 0
        for i, j in enumerate(pred):
            if pred[i] == test_tags[i]:
                k += 1
            al += 1
            total_number += 1
        print float(k/float(al)), "true"
        total_true += k
    print float(total_true / float(total_number)), "accu"

if __name__ == '__main__':
    # main()
    test()

