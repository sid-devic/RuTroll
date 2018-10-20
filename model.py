from process import create_data
import tensorflow as tf
import numpy as np
import random
import sklearn
from sklearn import preprocessing

# fixing seed for reproduction of results
np_rand = random.randint(0,10000)
from numpy.random import seed
seed(np_rand)

tf_rand = random.randint(0,10000)
from tensorflow import set_random_seed
set_random_seed(tf_rand)

print('np seed: ', np_rand)
print('tf seed: ', tf_rand)

# ========================================= #

iter_ = 30000
lr = 1e-2
batch_size = 512

# input dimensions are 2 x 3000 (unigram and bigram embedding)
input_dims = [None, 3000]
num_classes = 2

# input placeholder
X = tf.placeholder(tf.float32, input_dims)
x_flat = tf.contrib.layers.flatten(X)

# y_true
Y_ = tf.placeholder(tf.int32, [None, 2])

# entire model
fc1 = tf.layers.dense(x_flat, 1024)
fc2 = tf.layers.dense(fc1, 1024)
fc3 = tf.layers.dense(fc2, 512)
fc4 = tf.layers.dense(fc3, 256)
fc5 = tf.layers.dense(fc4, 256)
fc6 = tf.layers.dense(fc5, 128)
fc7 = tf.layers.dense(fc6, 64)
fc8 = tf.layers.dense(fc7, 32)
nn_out = tf.layers.dense(fc8, num_classes)

mse = tf.losses.mean_squared_error(Y_, nn_out)
cost = tf.reduce_mean(mse)
train = tf.train.GradientDescentOptimizer(lr).minimize(cost)

correct_prediction = tf.equal(tf.cast(tf.argmax(Y_, axis=1), tf.int64), tf.argmax(nn_out, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
   
    x_train, y_train, x_test, y_test = create_data()
    x_train = [a.A for a in x_train] 
    x_train = [a.reshape(-1) for a in x_train]
    x_test = [a.A for a in x_test]
    x_test = [a.reshape(-1) for a in x_test]

    print('Data snippit: ')
    #print(x_train[1:100])
    #print(y_train[1:100])
    print('\n')
 
    #y_test = sklearn.preprocessing.OneHotEncoder(y_test)

    for i in range(iter_):
        # sklearn shuffle to get minibatch
        shuffled_x, shuffled_y = sklearn.utils.shuffle(x_train, y_train, n_samples=batch_size)
        shuffled_y = np.reshape(shuffled_y, (batch_size, 1))
        one_hot_in = np.zeros((batch_size, 2))
        for element in range(batch_size):
            if shuffled_y[element] == 1:
                one_hot_in[element][1] = 1
            else:
                one_hot_in[element][0] = 1
        
        # train
        _, current_cost, acc = sess.run([train, cost, accuracy], feed_dict={X: shuffled_x, Y_: one_hot_in})
       
       # print training progress
        if i % 100 == 0:
            print('Iter: {0} cost: {1} Accuracy: {2}'.format(i, current_cost, acc)) 

    # test on validation set
    shuffled_x, shuffled_y = sklearn.utils.shuffle(x_test, y_test)
    shuffled_y = np.reshape(shuffled_y, (len(shuffled_y), 1))
    one_hot_in = np.zeros((len(shuffled_y), 2))
    
    for element in range(len(shuffled_y)):
        if shuffled_y[element] == 1:
            one_hot_in[element][1] = 1
        else:
            one_hot_in[element][0] = 1
     
    preds = sess.run([nn_out], feed_dict={X: shuffled_x, Y_: one_hot_in})
    correct = 0
    total = 0
    
    for j in range(len(preds[0])):
        if np.argmax(preds[0][j]) == np.argmax(one_hot_in[j]):
            correct += 1
        total += 1
    print('\nValidation acc: ', correct / total)

    # save model
    saver.save(sess, 'models/ten_layer_fc.ckpt')
