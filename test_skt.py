
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

    train_file = "gene-data1.txt"
    test_file = "gene-data1.txt"
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


# with open('stopwords.txt', 'r') as f:
#     stopwords = set([w.strip() for w in f])
comma_tokenizer = lambda x: nltk.word_tokenize(x)


def vectorize(train_words, test_words):
    # v = HashingVectorizer(tokenizer=comma_tokenizer, n_features=30000, non_negative=True)
    v1 = HashingVectorizer(tokenizer=comma_tokenizer, n_features=10000, non_negative=True)
    # v = TfidfVectorizer(tokenizer=comma_tokenizer, max_features=10000)
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
    classifier = LinearSVC(C=0.5, penalty="l2", dual=False)
    # classifier = LinearSVC()
    # random.seed(42)
    # classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10],
    # n_classes=3, batch_size=32, steps=10000, learning_rate=0.05)
    # classifier = SVC(kernel="linear")
    # classifier = MultinomialNB(alpha=0.000001)
    classifier.fit(train_data, train_tags)
    # classifier.fit(train_data.toarray(), train_tags)
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
    # if len(sys.argv) < 3:
    #     print '[Usage]: python classifier.py train_file test_file'
    #     sys.exit(0)
    train_file = "data.txt"
    test_file = "test.txt"
    train_words, train_tags, test_words, test_tags = input_data(train_file, test_file)
    # print test_tags
    # train_words, test_words, train_tags, test_tags = train_test_split(train_words, train_tags, test_size=0.30, random_state=42)
    # print test_tags
    # test_words = [u"在成都还能碰到屯中的男神也是6的不行所以来凑一场说走就走的旅行#都江堰##都江堰##都江堰#都江堰四川都江堰爆发泥石流，都江堰市中兴镇三溪村五里坡因山洪泥石流致11户人家被困或埋压，约40余人。接到报警后，成都消防指挥中心迅速调出十六中队、十九中队、都江堰江安路政府队共计4台消防车、24人赶往救援。话题详情关注"]
    train_data, test_data = vectorize(train_words, test_words)
    clf = train_svm(train_data, train_tags)
    # cPickle.dump(clf, open("svm.pkl", "wb"))
    pred = clf.predict(test_data)
    # print pred[0].encode('utf8')
    evaluate(numpy.asarray(test_tags), pred)
    # print pred
    # print test_tags
    k = 0
    for i, j in enumerate(pred):
        if pred[i].encode('utf8') == test_tags[i].encode('utf8'):
            k += 1
    print k
        # print i

def test():
    train_file = "3.25-data.txt"
    test_file = "test.txt"
    train_word, train_tag, test_words, test_tags = input_data_gen(train_file, test_file)
    # print test_tags
    # train_words, test_words, train_tags, test_tags = train_test_split(train_words, train_tags, test_size=0.30, random_state=42)
    # print test_tags
    # test_words = [u"在成都还能碰到屯中的男神也是6的不行所以来凑一场说走就走的旅行#都江堰##都江堰##都江堰#都江堰四川都江堰爆发泥石流，都江堰市中兴镇三溪村五里坡因山洪泥石流致11户人家被困或埋压，约40余人。接到报警后，成都消防指挥中心迅速调出十六中队、十九中队、都江堰江安路政府队共计4台消防车、24人赶往救援。话题详情关注"]
    total = 0
    all = 0.0
    for i in range(10):
        print i, "test"
        train_words, train_tags, test_words, test_tags = split_data(train_word, train_tag, i)
        # print len(train_words), len(test_words)
        train_data, test_data = vectorize(train_words, test_words)
        # train_data = numpy.array(train_data)
        train_tags = numpy.array(train_tags)
        # test_words = numpy.array(test_words)
        test_tags = numpy.array(test_tags)
        clf = train_svm(train_data, train_tags)
        # cPickle.dump(clf, open("svm.pkl", "wb"))
        pred = clf.predict(test_data.toarray())
        # print pred[0].encode('utf8')
        # evaluate(numpy.asarray(test_tags), pred)
        # print pred
        # print test_tags
        k = 0
        al = 0
        for i, j in enumerate(pred):
            if pred[i] == test_tags[i]:
                k += 1
            al += 1
            all += 1
        print float(k/float(al)), "true"
        total += k
    print float(total / float(all)), "accu"

if __name__ == '__main__':
    # main()
    test()
    # train_file = "train_file.txt"
    # test_file = "test_file.txt"
    # train_words, train_tags, test_words, test_tags = input_data(train_file, test_file)
    # # test_words = "我爱你"
    # train_data, test_data = vectorize(train_words, test_words)
    # clf = cPickle.load(open("svm.pkl", "rb"))
    # pred = clf.predict(test_data)
    # evaluate(numpy.asarray(test_tags), pred)