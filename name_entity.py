__author__ = 'bohaohan'
import nltk

sentence = "OR1-33 has an bad effect on OR2C3"
sent1 = nltk.word_tokenize(sentence)
sent2 = nltk.pos_tag(sent1)
sent3 = nltk.ne_chunk(sent2)
# print sent3.subtrees()
for i in sent3:
    # print i
    if isinstance(i, nltk.tree.Tree):
        for j in i:
            print j
#     for j in i:
#         print j
# nltk.tree.Tree