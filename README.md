# Text Mining and Sentiment Analysis [Python]

This project entails text mining of user-review data and sentiment analyses on a collection of reviews and accompanying star ratings from Yelp. The star ratings will be used here to indicate the sentiment label. For binary classification, we will need to convert the 1-5 scale rating values to {positive (1), negative (0)} values. The focus will be on:
1.	Examining the effectiveness of the dictionary: AFINN dictionary which includes words commonly used in user-generated content in the web.
2.	Develop and evaluate classification model using Naïve Bayes to help predict sentiment of the review: Bag of Words approach, with standard steps for creating the document term matrix (word vectors for each document; each row as document) - with term occurrences as values in the matrix.

Text cleaning steps using NLTK library:
- Tokenize
- Transform case to lower case
- Remove punctuation from each word
- Remove remaining token that are not alphabetic
- Filter out stop words
- Stemming 

Naïve Bayes model has been deployed using Flask as an interactive application.
Future Steps: Topic Modeling and Text Summarization

## Model Deployment to predict sentiment of the reviews using Python Flask
Positive Sentiment
![alt text](https://github.com/ishudewan/Text-Mining-and-Sentiment-Analysis/blob/master/Positive%20Sentiment.JPG)
Negative Sentiment
![alt text](https://github.com/ishudewan/Text-Mining-and-Sentiment-Analysis/blob/master/Negative%20Sentiment.JPG)
