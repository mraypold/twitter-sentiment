#!/usr/bin/env python

#
#   Author: Michael Raypold <mraypold@gmail.com>
#   April 29, 2016
#
#   Sentiment analysis for twitter.
#
#   Data set:
#   http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip
#
#   Further Resources:
#   https://www.youtube.com/watch?v=y3ZTKFZ-1QQ (Sentiment Classification)
#   http://pyvideo.org/video/2566/pickles-are-for-delis-not-software (Pickles)

import pandas as pd
import numpy as np
import sklearn.cross_validation
import sklearn.feature_extraction.text
import sklearn.metrics
import sklearn.naive_bayes
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    print '\nTwitter sentiment analysis\n'

    names = ['Sentiment', 'SentimentText']
    data = pd.read_csv("Sentiment_Analysis_Dataset.csv", header=0, \
                    delimiter=",", error_bad_lines=False)

    train, test = train_test_split(data, test_size=0.2)

    train_data = pd.DataFrame(train, columns=names)
    test_data = pd.DataFrame(test, columns=names)

    print "Tweets in training set: {}".format(train.shape[0])
    print "Tweets in test set: {}".format(test.shape[0])

    vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words='english')

    train_matrix = vectorizer.fit_transform(train_data['SentimentText'])
    test_matrix = vectorizer.transform(test_data['SentimentText'])

    positive_cases_train = (train_data['Sentiment'] == 1)
    positive_cases_test = (test_data['Sentiment'] == 1)

    classifier = sklearn.naive_bayes.MultinomialNB()
    classifier.fit(train_matrix, positive_cases_train)

    predicted_sentiment = classifier.predict(test_matrix)
    predicted_probs = classifier.predict_proba(test_matrix)

    accuracy = classifier.score(test_matrix, positive_cases_test)
    precisions, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        positive_cases_test, predicted_sentiment)

    print '\n'
    print 'Accuracy = {}'.format(accuracy)
    print 'Precision = {}'.format(precisions)
    print 'Recall = {}'.format(recall)
    print 'F1 Score = {}'.format(f1)
    print '\n'

    print 'Saving classifier to pickel...'
    joblib.dump(classifier, 'classifier.pkl', compress=9)

    print 'Saving vectorizer to pickel...'
    joblib.dump(vectorizer, 'vectorizer.pkl', compress=9)

    print 'Pickling completed!'
