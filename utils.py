from data import twitter_samples
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
