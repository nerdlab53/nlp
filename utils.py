import matplotlib.pyplot as plt
import random
import re 
import string
import nltk as nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

nltk.download('twitter_samples')

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Splitting the tweets into positive and negative.
def split_tweets():
    sentiments = {'positive', 'negative'}
    negative_tweets = []
    positive_tweets = []
    for sentiment in sentiments:
        if sentiment == 'positive':
            positive_tweets = twitter_samples.strings('positive_tweets.json')
        else : 
            negative_tweets = twitter_samples.strings('negative_tweets.json')
    return positive_tweets, negative_tweets
