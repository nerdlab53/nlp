import numpy as np
from data import twitter_samples
# Splitting the tweets into positive and negative.
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

