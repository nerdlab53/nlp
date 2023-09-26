import re
import string
import numpy as np
from data import twitter_samples
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

def split_tweets():
    '''

    Splitting the positive and negative tweets
    
    '''

    sentiments = {'positive', 'negative'}
    negative_tweets = []
    positive_tweets = []
    for sentiment in sentiments:
        if sentiment == 'positive':
            positive_tweets = twitter_samples.strings('positive_tweets.json')
        else : 
            negative_tweets = twitter_samples.strings('negative_tweets.json')
    return positive_tweets, negative_tweets

def sigmoid(z):
    '''

    Find the sigmoid value for an input z

    '''

    h = 1./(1 + np.exp(-z))

    return h


def gradientDescent(x, y, theta, alpha, num_iters, m):
    '''
    
    Computes cost and optimizes the weights
    
    '''
    for i in range(0, num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        J = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1-y.T), np.log(1-(h))))
        theta = theta - (alpha/m) * np.dot(x.T, (h-y))
    J = float(J)
    return J, theta


def extract_features(tweet, freqs, process_tweet=process_tweet):
    '''
    Extracts features
    
    '''
    word_l = process_tweet(tweet)
    x = np.zeros(3)
    x[0] = 1
    for word in word_l:
        x[1] += freqs.get((word, 1.0), 0)
        x[2] += freqs.get((word, 0.0), 0)
    x = x[None,:]
    assert(x.shape==(1,3))
    return x

def predict_tweet(tweet, freqs, theta):
    '''
    Predicts a tweet's sentiment

    '''
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred

def test_logistic_regression(test_x, test_y, freqs, theta, predict_tweet=predict_tweet):
    '''
    Trains logistic regression on the data
    
    '''
    y_hat = []
    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)
        if y_pred > 0.5:
            y_hat.append(1.0)
        else:
            y_hat.append(0.0)
    accuracy = (y_hat == np.squeeze(test_y)).sum()/len(test_x)

    return accuracy

