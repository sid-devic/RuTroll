# Russian Tweet Classifier
This is a twofold project. The main idea is to create a chrome extension which highlights tweets that are likely coming from the [Russian political propaganda machine](https://www.cnn.com/2018/10/19/politics/russian-troll-instructions/index.html). The secondary project was to develop an in-house machine learning algorithm to classify and differentiate Russian tweets from these troll farms.  
  
The bulk of our data comes from [the website fivethirtyeight](https://github.com/fivethirtyeight/russian-troll-tweets), which released a dataset of over 1.4 million tweets known to have come from botnets or Russian trolls. For normal, non-malicious tweet data, we used [kaggle's sentiment analysis dataset](https://www.kaggle.com/c/twitter-sentiment-analysis2).  
  
Our model is a ten-layer fully connected neural network built with the abstracted ```tf.layers``` api, with the following layer widths: ```[1024, 1024, 512, 256, 256, 128, 64, 32]``` which is motivated by [1].  
  
## Preprocessing


## References
[1] Rudolph, S. (1997). On topology, size and generalization of non-linear feed-forward neural networks. Neurocomputing, 16(1), pp.1-22.  
