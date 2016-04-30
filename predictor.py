#!/usr/bin/env python

#
#   Author: Michael Raypold <mraypold@gmail.com>
#   April 30, 2016
#
#   Sentiment analysis for twitter.
#
#   Resources:
#   http://stackoverflow.com/questions/25788151/bringing-a-classifier-to-production
#   https://blog.scrapinghub.com/2014/03/26/optimizing-memory-usage-of-scikit-learn-models-using-succinct-tries/
#
#   How to use:
#   - Requests stdin and then output whether the text is positive or negative
#   sentiment.
#   - Trained on twitter data, so stick to strings less than 140 characters.
#

import pandas as pd
import numpy as np
import sklearn.cross_validation
import sklearn.feature_extraction.text
import sklearn.metrics
import sklearn.naive_bayes
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

print 'Loading vectorizer and classifier from pickles.'
print 'This will take a couple seconds...'
vectorizer = joblib.load('vectorizer.pkl')
classifier = joblib.load('classifier.pkl')
print 'Loading done!'

def predict(text):
    '''Takes a string of text and returns whether True for positive sentiment'''
    x = vectorizer.transform([text])
    return classifier.predict(x)

if __name__ == '__main__':
    while True:
        text = raw_input('Enter tweet text: ')
        positive = predict(text)
        if positive:
            print 'Positive sentiment'
        else:
            print 'Negative sentiment'
