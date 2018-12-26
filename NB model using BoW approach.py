#Import libraries
from flask import Flask, render_template, abort, request
import sys
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import numpy as np
#Import dataset
dataset = pd.read_csv('data.csv', sep= ';', encoding = 'ISO-8859-1')
#Clean the text
def text_cleaning(text):
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    return stemmed
#Subset the dataset into binary values
dataset_pn = dataset[(dataset['stars'] == 1) | (dataset['stars'] == 5)]

X = dataset_pn['text']
y = dataset_pn['stars']

#Bag of Words approach - creating the document matrix with term occurrences as the values
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer=text_cleaning).fit(X)
X = cv.transform(X)
#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
#Building the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
#Making the predictions
preds = nb.predict(X_test)
#Creating the confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))


