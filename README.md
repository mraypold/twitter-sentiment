### Twitter Sentiment Analysis

Classifies tweets as either having positive or negative sentiment.

This is a really basic introduction to SA. For a more interesting approach see the resources section where tweets are classified using emoticons and then trained using supervised learning.

Also note that tweets in this case are either labeled as positive or negative. In reality a tweet could be neutral (eg: the person is stating a fact). Once again, see the resources section for a more thorough overview.

Finally, the accuracy (with the current implementation) is about 77% for the tested dataset. However, I have found this did not translate very well to real world testing.

### Files

* `sentiment.py` trains the data.
* `predictor.py` request input from stdin and returns the sentiment.

### Installing

I have not uploaded the python pickles or the training/testing data because they are too large for a repository.

[Here](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/) is a description of the dataset and you can download it [here](http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip).

I have renamed the file from `Sentiment Analysis Dataset.csv` to `Sentiment_Analysis_Dataset.csv`.

This was run with python 2.7, but should easily convert to 3.4 if needed by altering the print statements and input mechanisms.

The standard pandas, numpy and scikit-learn libraries are needed.

### Resources
* [Ryan Rosario - Sentiment Classification Using scikit-learn](https://www.youtube.com/watch?v=y3ZTKFZ-1QQ) - You will find a description for much of the code in `sentiment.py` in this talk.
* [Twitter Sentiment Classification using Distance Supervision - pdf](https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf) - A research paper for classifying the sentiment of Twitter messages.
* [Optimizing Memory Usage of Scikit-Learn Models Using Succinct Tries](https://blog.scrapinghub.com/2014/03/26/optimizing-memory-usage-of-scikit-learn-models-using-succinct-tries/) - I did not make use of this, but it is still an interesting read.
* [NLP for Movie Reviews](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words)

### License

Some of the code is from Ryan Rosario's (Facebook) talk at PyData (see Resources).

The remainder is licensed under MIT (see LICENSE).
