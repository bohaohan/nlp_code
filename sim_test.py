
# coding: utf-8

import jieba

__author__ = 'bohaohan'
import gensim
# from passage.preprocessing import Tokenizer
from keras.preprocessing.text import Tokenizer, one_hot, base_filter
# # model = gensim.models.Word2Vec.load_word2vec_format("1.vector", binary=False)
# tokenizer = Tokenizer(nb_words=1000)
# # a = ["的 一个 有趣 发展 是 进行 核外扩展 的 能力.", "数据流 还是"]
a  ="的 一个 有趣 发展 是 进行 核外扩展 的 能力."
# a = ["a d d", "d a"]
# a = ["我是一个爱生活的人", "他也是一个爱生活的人"]
# one_h = one_hot(filters=base_filter(), n=30, text=a)
# # o.fit_on_texts(a)
# # b = one_h(a)
# print one_hot(filters=base_filter(), n=30, text=a)
# print one_hot(filters=base_filter(), n=30, text=a)

# a=['hello world', 'foo bar']
# tokenizer = Tokenizer()
# train_tokens = tokenizer.fit_transform(a)
# print train_tokens
# comma_tokenizer = lambda x: jieba.cut(x, cut_all=True)
# from sklearn.feature_extraction.text import HashingVectorizer
# v = HashingVectorizer(tokenizer=comma_tokenizer, n_features=30000, non_negative=True)
# train_data = v.fit_transform(a)
# print train_data

# import jieba
a = "我是一个男孩"
c = jieba.cut(a, cut_all=False)
w = ""
# print(", ".join(c))
for i in c:
    w += i + " "
    # print i
w = w[:len(w)-1].encode('utf8')
# w = "我 是 一个男孩"
print one_hot(filters=base_filter(), n=30000, text=w)
# print w
# # print c.next()