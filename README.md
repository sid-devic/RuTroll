# Russian Tweet Classifier
Change 'dataset' variable in process.py (main) to the location of the folder containing all the russian tweet csv from [here](https://github.com/fivethirtyeight/russian-troll-tweets)    

To load vectorizer.pickle: follow [this](https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict) to load the vectorizer, and then call 'vectortweet = vectorizer.transform([tweet])' where tweet is a string.
