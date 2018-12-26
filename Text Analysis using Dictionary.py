#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import numpy as np

#Import dataset
dataset = pd.read_csv('data.csv', sep= ';', encoding = 'ISO-8859-1')
dict_affin_pos = pd.read_csv('Affin Positive.txt', sep= '\n', encoding = 'ISO-8859-1')
dict_affin_neg = pd.read_csv('Affin Negative.txt', sep= '\n', encoding = 'ISO-8859-1')

positive_vocab = list(dict_affin_pos.iloc[:,0])
negative_vocab = list(dict_affin_neg.iloc[:,0])

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

#Calculate score based on the dictionary
def calculate_pos_score(text):
    reviewTokens = text_cleaning(text)
    numPosWords = 0
    for word in reviewTokens:
        if word in positive_vocab:
            numPosWords = numPosWords + 1
    numNegWords = 0
    for word in reviewTokens:
        if word in negative_vocab:
            numNegWords = numNegWords + 1
    score = numPosWords - numNegWords
    if score > 1:
        score = 5
    else:
        score = 1
    return score

#Subset the dataset into binary values
dataset_pn = dataset[(dataset['stars'] == 1) | (dataset['stars'] == 5)]

X = dataset_pn['text']
y = dataset_pn['stars']

scored_dataset = pd.DataFrame()

#Calculate score for all reviews

for i in range(len(X)):
    text = X.iloc[i]
    text = text_cleaning(text)
    score = calculate_pos_score(text)
    scored_dataset.loc[i,0] = text
    scored_dataset.loc[i,1] = score

#Check against the dataset if it's correct or incorrect and calculate the accuracy
for i in range(len(y)):
    rating = y.iloc[i]
    scored_rating = scored_dataset.loc[i,1]
    if rating == scored_rating:
        scored_dataset.loc[i,2] = "Correct"
    else:
        scored_dataset.loc[i,2] = "Incorrect"
#Calculate accuracy
accuracy_dict = ((len(scored_dataset[scored_dataset[2] == 'Correct']))/(len(scored_dataset)))*100
print(accuracy_dict)



