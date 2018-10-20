from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer
import time
import csv

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
            tweet_str = [word for word in tokenized_tweet if word.isalpha()]
            rut.content = ' '.join(tweet_str)
    
            # TODO: add stop word removal, or maybe not. Stop words can tell us some information.

            tweets.append(rut)

    #tweets = tweets[:100000]

    return tweets

# TODO count RightTroll and LeftTroll, load all files and see how much data we end up with.
# 

def main():
    start = time.time()
    dataset_dir = '/mnt/c/Users/sidda/Documents/Programming/datasets/russian-troll-tweets/'
    
    corpus = []
    n_grams = []
    labels = []

    left_count = 0
    right_count = 0

    for index in range(3,5):
        rut = load_tweets(dataset_dir + 'IRAhandle_tweets_{0}.csv'.format(index))
        
        for tweet in rut:
            corpus.append(tweet.content)
            if tweet.account_category == 'LeftTroll':
                labels.append(0)
                left_count += 1
            else:
                labels.append(1)
                right_count += 1
            #print(tweet.content)
            #print(tweet.account_category)
 
    vectorizer = CountVectorizer(ngram_range=(1,2), max_features=10000)
    word_vec = vectorizer.fit_transform(corpus).todense() 
 
    for tweet in corpus:
        n_grams.append(vectorizer.transform([tweet]))

    end = time.time()
    print('Left trolls:', left_count, 'Right trolls:', right_count)
    print('Load time:', end - start)
    
    # write to csv
    with open('tweets.csv', 'w') as csvfile:
        fieldnames = ['label', 'n_grams']
        writer = csvDictWriter(csvfile, fieldnames=fieldnames)
        
        for embedding in n_grams:
            writer.writerow({'label': 1, 'n_grams': embedding})

    csvfile.close()

if __name__ == "__main__":
    main()
