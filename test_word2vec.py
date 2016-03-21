# coding: utf-8
__author__ = 'bohaohan'
from os import path
import gensim


_W2V_BINARY_PATH = path.dirname(path.abspath(__file__)) + "/word2vec/GoogleNews-vectors-negative300.bin"
model = gensim.models.Word2Vec.load_word2vec_format(_W2V_BINARY_PATH, binary=True)
positive = model['positive']
print positive
print len(positive)
