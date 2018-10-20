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

# for abstracting model creation
arch = [256, 128, 128, 64, 32, 32]
print_weights = False
normalize_data = False

iter_ = 50000
lr = 1e-1
batch_size = 32

def fc(x, input_size, output_size, layer_num, softmax=True):
    var_name = 'W_' + str(layer_num)
    W = tf.Variable(tf.truncated_normal([input_size, output_size]), name=var_name)

    bias_name = 'B_' + str(layer_num)
    b = tf.Variable(tf.truncated_normal([output_size]), name=bias_name)

    if softmax:
        return tf.nn.relu(tf.matmul(x, W) + b)
    else:
        return tf.matmul(x, W) + b

def fcnn(input_data, input_dims, arch, num_classes=2):
    # first layer
    nn = fc(x=input_data, 
            input_size=input_dims,
            output_size=arch[0], 
            layer_num=1)

    # hidden layers
    for layer in range(1, len(arch)):
        nn = fc(x=nn, 
                input_size=arch[layer-1], 
                output_size=arch[layer], 
                layer_num=layer + 1)

    # output layer, don't use softmax
    nn = fc(x=nn, 
            input_size=arch[len(arch) - 1], 
            output_size=num_classes, 
            layer_num='out', 
            softmax=False)

    return nn

# input dimensions are 2 x 3000 (unigram and bigram embedding)
input_dims = [None, 1, 3000]
num_classes = 2

# input placeholder
X = tf.placeholder(tf.float32, input_dims)
x_flat = tf.contrib.layers.flatten(X)

# y_true
Y_ = tf.placeholder(tf.int32, [None, 1])

y_true_cls = tf.argmax(tf.one_hot(Y_, 2), axis=1)

# entire model
#nn_out = fcnn(x_flat, 6000, arch)
#nn_out = tf.argmax(nn_out, axis=1)
nn_out = tf.layers.dense(x_flat, num_classes)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true_cls, logits=nn_out)
cost = tf.reduce_mean(cross_entropy)
train = tf.train.AdamOptimizer(lr).minimize(cost)

#correct_prediction = tf.equal(y_true_cls, nn_out)
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
   
    x_train, y_train, x_test, y_test = create_data()
   
    print('Data snippit: ')
    print(x_train[1])
    print(y_train[1])
    print('\n')
 
    y_test = sklearn.preprocessing.OneHotEncoder(y_test)

    for i in range(iter_):
        # sklearn shuffle to get minibatch
        shuffled_x, shuffled_y = sklearn.utils.shuffle(x_train, y_train, n_samples=batch_size)
        
        print(np.shape(shuffled_x[1]))
        print(np.shape(shuffled_y[0]))

        # train
        _, current_cost = sess.run([train, cost], feed_dict={X: shuffled_x, Y_: shuffled_y})
        accuracy = 0 
        # print training progress
        if i % 100 == 0:
            print('Iter: {0} Train mse: {1} Accuraccy: {2}'.format(i, current_cost, accuracy))
   
    '''
    # ======================================================================= #
    # print MSE on entire training set
    print('\nTesting on train-set...')
    pred = sess.run(icnn_out, feed_dict={X: x_train})
    error = []

    #print('{0:25} {1}'.format('pred', 'real'))
    for j in range(len(pred)):
        #print('{0:<25} {1}'.format(str(pred[j][0]), y_test[j]))
        error.append((y_train[j] - pred[j][0]) ** 2)

    print('\nTotal train size: ', len(y_train))
    print('Train mse: ', sum(error) / len(error))

    # ======================================================================== #
    # test on validation set
    print('\nTesting on validation-set...')
    pred = sess.run(icnn_out, feed_dict={X: x_test})
    error = []

    #print('{0:25} {1}'.format('pred', 'real'))
    for j in range(len(pred)):
        #print('{0:<25} {1}'.format(str(pred[j][0]), y_test[j]))
        error.append((y_test[j] - pred[j][0]) ** 2)

    print('\nTotal test size: ', len(y_train))
    print('Test mse: ', sum(error) / len(error))

    print('\nNormalized Data:', normalize_data)
    print('Architecture:',icnn_arch)
    print('lr_:',lr)
    print('iterations:',iter_)
    '''
