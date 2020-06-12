# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 23:19:05 2020

@author: ssark
"""

import spacy
nlp = spacy.load('en_core_web_sm')
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import string
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)

# load the model from disk
filename = 'Classifier.pkl'
clf = pickle.load(open(filename, 'rb'))

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



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = text_data_clean(message)
        my_prediction = clf.predict(data)
    return render_template('result.html',prediction = my_prediction)
        

		



if __name__ == '__main__':
	app.run(debug=True)