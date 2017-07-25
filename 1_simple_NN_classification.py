# coding:utf-8
'''
python 3.5
tensorflow: 1.2.0
author: helloholmes
'''

import tensorflow as tf
import input_data_mnist
import numpy as np
import matplotlib.pyplot as plt
import random
# 读取mnist数据集，标签为独热码形式，
mnist = input_data_mnist.read_data_sets('MNIST_data', one_hot = True)

# train dataset images shape: [55000, 784], labels shape: [55000, 10]
# print(mnist.train.images.shape, mnist.train.labels.shape)
# test dataset images shape: [10000, 784], labels shape: [10000, 10]
# print(mnist.test.images.shape, mnist.test.labels.shape)

# hyper parameters
BATCH_SIZE = 50
Learning_Rate = 0.001
Hidden = 1024
validate_len = 1000
test_len = 2000

#validate dataset
x_val = mnist.test.images[:validate_len]
y_val = mnist.test.labels[:validate_len]

# show one image
# plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap = 'gray')
# plt.title('%i' % np.argmax(mnist.train.labels[0]))
# plt.show()

x_ = tf.placeholder(dtype = tf.float32, shape = [None, 28 * 28])
y_ = tf.placeholder(dtype = tf.float32, shape = [None, 10])

# full connect neural networks
# the number of hidden neural cells is Hidden
# activation is relu/sofxmax/tanh
# the nn shape: [None, Hidden]
nn = tf.layers.dense(
	inputs = x_,
	units = Hidden,
	activation = tf.nn.relu,
	use_bias = True,
	kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01),
	bias_initializer = tf.constant_initializer(0.1),
	name = 'hidden_layer',
	reuse = None)
# print(nn.get_shape().as_list())

# output shape: [None, 10]
# activation is softmax(classification)
output = tf.layers.dense(
	inputs = nn,
	units = 10,
	activation = tf.nn.softmax,
	kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01),
	bias_initializer = tf.constant_initializer(0.1),
	name = 'output_layer',
	reuse = None)
# choose the biggest probability as the predict label
predict_label = tf.argmax(output, axis = 1)

# define the loss
# cross entropy: -sum(y_ * log(output)) 交叉熵
loss = tf.losses.softmax_cross_entropy(onehot_labels = y_, logits = output)

# define the train operation
# AdamOptimizer/RMSPropOptimzier/AdadeltaOptimizer/Adagrad/Momentum/GradientFescent
train_op = tf.train.AdamOptimizer(Learning_Rate).minimize(loss)

# define the accuracy compute operation
# return (accuracy, update_op)
accuracy = tf.metrics.accuracy(
	labels = tf.argmax(y_, axis = 1), predictions = predict_label,)[1]

#initialize the global and local variables
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# the graph has been created above
# ---------------------------------- #
sess = tf.Session()
# computing started
# step 1: initialize variables in graph
sess.run(init_op)

# step 2: train
for i in range(1000):
	# load the training data
	x_train, y_train = mnist.train.next_batch(BATCH_SIZE)
	# feed the training data into graph ,train and compute loss
	_, loss_ = sess.run([train_op, loss], feed_dict = {x_: x_train, y_: y_train})

	if i % 50 == 0:
		accuracy_ = sess.run([accuracy], feed_dict = {x_: x_val, y_: y_val})
		print('step', i,': loss: %.4f' % loss_, 'validate accuracy: %.2f' % accuracy_[0])

	# save the network

# step 3: test

x_test = mnist.test.images[:-test_len]
y_test = mnist.test.labels[:-test_len]
test_acc = sess.run([accuracy], feed_dict = {x_: x_test, y_:y_test})
print('test accuracy: %.4f' % test_acc[0])