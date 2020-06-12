# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 19:10:42 2020

@author: ssark
"""
import spacy


nlp = spacy.load('en_core_web_sm')

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
import string
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
import pickle

data_rest = pd.read_csv("Restaurant_Reviews.tsv", sep='\t')

data_imdb = pd.read_csv("Imdb_Reviews.txt", sep='\t')

data_imdb.columns = ['Review','Liked']

data = data_rest.append(data_imdb, ignore_index=True)

def text_data_clean(sent):
    doc = nlp(sent)
    
    tokens = []
    for token in doc:
        if token.lemma_ !="-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
            
        tokens.append(temp)
    
    cleaned_tokens = []
    
    for token in tokens:
        if token not in stopwords and token not in string.punctuation:
            cleaned_tokens.append(token)
    
    return cleaned_tokens

data['Review'] = data['Review'].apply(lambda x: text_data_clean(x))
data['Review'] = data['Review'].apply(lambda x: " ".join(x))

tfidf = TfidfVectorizer(tokenizer= None)
classifier = LinearSVC()
X = data['Review']
y = data['Liked']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state =0)

clf = Pipeline([('tfidf',tfidf),('clf',classifier)])

clf.fit(X_train,y_train)

pickle.dump(clf, open('Classifier.pkl', 'wb'))







