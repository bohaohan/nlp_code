__author__ = 'zhengxiaoyu'
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
import csv
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
#nltk.download()

# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%

traindf = pd.read_json(".../train.json")
traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients'] if z!='salt' and z!='water']
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients'] if z!='salt' and z!='water']

testdf = pd.read_json(".../test.json")
testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients'] if z!='salt' and z!='water']
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients'] if z!='salt' and z!='water']



corpustr = traindf['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word",
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
tfidftr=vectorizertr.fit_transform(corpustr)

vectorizerhas = HashingVectorizer(stop_words='english',
                             ngram_range = ( 1 , 2 ),analyzer="word" )
hastr = vectorizerhas.fit_transform(corpustr)

corpusts = testdf['ingredients_string']

vectorizerts = TfidfVectorizer(stop_words='english')
tfidfts=vectorizertr.transform(corpusts)
vectorizerhast = HashingVectorizer(stop_words='english')
hasts = vectorizerhast.transform(corpusts)

predictors_tr = tfidftr
targets_tr = traindf['cuisine']

predictors_ts = tfidfts


classifier = LinearSVC(C=0.5, penalty="l2", dual=False)
#parameters = {'C':[0.5, 10]}
#clf = LinearSVC()
#clf = LogisticRegression()

#classifier = grid_search.GridSearchCV(classifier, parameters)

classifier.fit(predictors_tr,targets_tr)

predictions=classifier.predict(predictors_ts)
testdf['cuisine'] = predictions
#testdf = testdf.sort('id' , ascending=True)

testdf[['id' , 'cuisine' ]].to_csv("submission.csv",index = False)



reader = csv.reader(open("submission.csv"))