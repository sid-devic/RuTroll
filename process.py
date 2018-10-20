from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer
import time
import csv
import pandas as pd
import pickle
import numpy as np
from sklearn.utils import shuffle

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

def load_normal_tweets(path, max_tweets=50000):
    tweets = []
    counter = 0
    f = pd.read_csv(path, encoding='latin1')
    df = f.values.tolist()
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

    for x in range(len(f)):
        tweet = df[x][len(df[x]) - 1]

        # clean up the tweet, removing non standard chars
        tweet = tokenizer.tokenize(tweet)
        tweet = [word for word in tweet if word.isalpha()]
        tweet = ' '.join(tweet)

        tweets.append(tweet)

        counter += 1
        if counter > max_tweets:
            break

    # we only care for about 250k tweets (balancing with our russian ones)
    return tweets

def create_data(write=False):
    print('Loading russian tweets into memory...')
    start = time.time()
    dataset_dir = '/home/sid/datasets/tweets/russian-troll-tweets/'
    
    corpus = []
    n_grams = []
    labels = []

    left_count = 0
    right_count = 0

    exit_flag = False

    for index in range(3,8):
        # exit after 250k troll tweets parsed
        if exit_flag:
            break

        rut = load_tweets(dataset_dir + 'IRAhandle_tweets_{0}.csv'.format(index))
        
        for tweet in rut:
            if len(labels) > 50000:
                exit_flag = True
                break

            corpus.append(tweet.content)
            labels.append(1)
            if tweet.account_category == 'LeftTroll':
                left_count += 1
            else:
                right_count += 1
            #print(tweet.content)
            #print(tweet.account_category)
        
    print('Loading non-troll tweets into memory...')

    # load non_troll tweets
    non_troll_path = '/home/sid/datasets/tweets/non-russian/non-russian.csv'
    norm_tweets = load_normal_tweets(non_troll_path)

    # append 0's to the labels for each normal tweet
    for j in range(len(norm_tweets)):
        labels.append(0)

    # concat our list of normal tweets
    corpus += norm_tweets
    
    if write:
        # write to csv
        with open('tweets.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            
            for i in range(len(corpus)):
                append_label = ''
                if labels[i] == 1:
                    append_label = 'russian'
                else:
                    append_label = 'non-russian'
                writer.writerow([corpus[i], append_label])

        csvfile.close()

    vectorizer = CountVectorizer(max_features=3000)
    word_vec = vectorizer.fit_transform(corpus).todense() 
 
    with open('vectorizer.pickle', 'wb') as handle:
        pickle.dump(word_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for tweet in corpus:
        n_gram_tweet = vectorizer.transform([tweet])
        #print(np.shape(n_gram_tweet))
        n_grams.append(n_gram_tweet)

    end = time.time()
    print('Left trolls:', left_count, 'Right trolls:', right_count)
    print('Load time:', end - start) 
    print('Total tweets:', len(n_grams))
 
    # shuffle it up, train/test split
    n_grams, labels = shuffle(n_grams, labels)
    tr_test = 25000
    train_x = n_grams[:tr_test]
    train_y = labels[:tr_test]

    test_x = n_grams[tr_test:]
    test_y = labels[tr_test:]

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def main():
    create_data(write=True)

if __name__ == '__main__':
    main()
