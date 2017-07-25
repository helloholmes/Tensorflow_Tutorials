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

# hyper parameters
BATCH_SIZE = 50
Learning_Rate = 0.001
Hidden = 1024
validate_len = 1000
test_len = 2000

#validate dataset
x_val = mnist.test.images[:validate_len]
y_val = mnist.test.labels[:validate_len]

x_ = tf.placeholder(dtype = tf.float32, shape = [None, 28 * 28])
# the input image should reshape into 3D because of convolution
# the reshaped shape: [batch, height, width, channel]
x_image = tf.reshape(x_, [-1, 28, 28, 1]) 
y_ = tf.placeholder(dtype = tf.float32, shape = [None, 10])

# first convolution layer
# the filter shape: [5, 5, 1, 32]
# stride = [1, 1, 1, 1]
# conv1 shape: [-1, 28, 28, 32]
conv1 = tf.layers.conv2d(
	inputs = x_image,
	filters = 32,
	kernel_size = 5,
	strides = 1,
	padding = 'SAME',
	activation = tf.nn.relu,
	kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01),
	bias_initializer = tf.constant_initializer(0.1),
	name = 'conv1')
# print(conv1.get_shape().as_list())

# first pool layer
# 2 x 2 pool
# pool1 shape: [-1, 14, 14, 32]
pool1 = tf.layers.max_pooling2d(
	inputs = conv1,
	pool_size = 2,
	strides = 2,
	padding = 'SAME',
	name = 'pool1')
# print(pool1.get_shape().as_list())

# second convolution layer
# the filter shape: [5, 5, 32, 64]   strides shape: [1, 1, 1, 1]
# conv2 shape: [-1, 14, 14, 64]
conv2 = tf.layers.conv2d(
	inputs = pool1,
	filters = 64,
	kernel_size = 5,
	strides = 1,
	padding = 'SAME',
	activation = tf.nn.relu,
	kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01),
	bias_initializer = tf.constant_initializer(0.1),
	name = 'conv2')

# second pool layer
# 2 x 2 pool
# pool2 shape: [-1, 7, 7, 64]
pool2 = tf.layers.max_pooling2d(
	inputs = conv2,
	pool_size = 2,
	strides = 2,
	padding = 'SAME',
	name = 'pool2')

# flatten the 2-D image into 1-D
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

# full connect layer
# fc shape: [-1, Hidden]
fc = tf.layers.dense(
	inputs = pool2_flat,
	units = Hidden,
	activation = tf.nn.relu,
	use_bias = True,
	kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01),
	bias_initializer = tf.constant_initializer(0.1),
	name = 'fc')

output = tf.layers.dense(
	inputs = fc,
	units = 10,
	activation = tf.nn.softmax,
	kernel_initializer  =tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01),
	bias_initializer = tf.constant_initializer(0.1),
	name = 'output')

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
for i in range(5000):
	# load the training data
	x_train, y_train = mnist.train.next_batch(BATCH_SIZE)
	# feed the training data into graph ,train and compute loss
	_, loss_ = sess.run([train_op, loss], feed_dict = {x_: x_train, y_: y_train})

	if i % 50 == 0:
		accuracy_ = sess.run([accuracy], feed_dict = {x_: x_val, y_: y_val})
		print('step', i,': loss: %.4f' % loss_, 'validate accuracy: %.2f' % accuracy_[0])

	# save the network

# step 3: test

x_test = mnist.test.images[:test_len]
y_test = mnist.test.labels[:test_len]
test_acc = sess.run([accuracy], feed_dict = {x_: x_test, y_:y_test})
print('test accuracy: %.4f' % test_acc[0])