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
    with open(fn, 'r') as f:
        for line in f.readlines():
            fields = line.split(',')
            rut = Rutweet(fields[0], fields[1], fields[2],
                          fields[3], fields[4], fields[5],
                          fields[6], fields[7], fields[8],
                          fields[9], fields[10], fields[11],
                          fields[12], fields[13], fields[14])
            tweets.append(rut)

    tweets = tweets[:10000]

    return tweets


def main():
    dataset_dir = '/mnt/c/Users/sidda/Documents/Programming/datasets/russian-troll-tweets/'
    rut = load_tweets(dataset_dir + 'IRAhandle_tweets_11.csv')

    for tweet in rut:
        print(tweet.content)


if __name__ == "__main__":
    main()