from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer

class Rutweet:
    def __init__(self, external_author_id, author, content,
                region, language, publish_date, harvested_date,
                following, followers, updates, post_type, account_type,
                retweet, account_category, new_june_2018):
    
        self.external_author_id = external_author_id
        self.author = author
        self.content = content
        self.region = region
        self.language = language
        self.publish_date = publish_date
        self.harvested_date = harvested_date
        self.following = following
        self.updates = updates
        self.post_type = post_type
        self.account_type = account_type
        self.retweet = retweet
        self.account_category = account_category
        self.new_june_2018 = new_june_2018


def load_tweets(fn):
    tweets = []
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
   
    with open(fn, 'r') as f:
        for line in f.readlines():
            fields = line.split(',')
            rut = Rutweet(fields[0], fields[1], fields[2],
                          fields[3], fields[4], fields[5],
                          fields[6], fields[7], fields[8],
                          fields[9], fields[10], fields[11],
                          fields[12], fields[13], fields[14])

            # only parse english troll tweets
            if rut.language != 'English':
                continue

            if rut.account_category != 'RightTroll' and rut.account_category != 'LeftTroll':
                continue
            
            # now clean up the tweet a bit
            tokenized_tweet = tokenizer.tokenize(rut.content)
            tweet_str = [str.lower(word) for word in tokenized_tweet if word.isalpha()]
            rut.content = ' '.join(tweet_str)
    
            # TODO: add stop word removal, or maybe not. Stop words can tell us some information.

            tweets.append(rut)

    tweets = tweets[:10000]

    return tweets

# TODO count RightTroll and LeftTroll, load all files and see how much data we end up with.
# 

def main():
    dataset_dir = '/mnt/c/Users/sidda/Documents/Programming/datasets/russian-troll-tweets/'
    rut = load_tweets(dataset_dir + 'IRAhandle_tweets_3.csv')
    corpus = []

    for tweet in rut:
        corpus.append(tweet.content)
        #print(tweet.content)
        #print(tweet.account_category)

    vectorizer = CountVectorizer(ngram_range=(1,2), max_features=10000)
    word_vec = vectorizer.fit_transform(corpus).todense()
    x = vectorizer.transform(['confirms trump boi i\'m edgy twitter liberal']).toarray()

if __name__ == "__main__":
    main()
