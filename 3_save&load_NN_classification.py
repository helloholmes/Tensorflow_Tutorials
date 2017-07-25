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
save_network = True
load_network = True

x_val = mnist.test.images[:validate_len]
y_val = mnist.test.labels[:validate_len]

def train():
	x_ = tf.placeholder(dtype = tf.float32, shape = [None, 28 * 28])
	y_ = tf.placeholder(dtype = tf.float32, shape = [None, 10])

	nn = tf.layers.dense(
		inputs = x_,
		units = Hidden,
		activation = tf.nn.relu,
		use_bias = True,
		kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01),
		bias_initializer = tf.constant_initializer(0.1),
		name = 'hidden_layer',
		reuse = None)

	output = tf.layers.dense(
		inputs = nn,
		units = 10,
		activation = tf.nn.softmax,
		kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01),
		bias_initializer = tf.constant_initializer(0.1),
		name = 'output_layer',
		reuse = None)

	predict_label = tf.argmax(output, axis = 1)

	loss = tf.losses.softmax_cross_entropy(onehot_labels = y_, logits = output)

	train_op = tf.train.AdamOptimizer(Learning_Rate).minimize(loss)

	accuracy = tf.metrics.accuracy(
		labels = tf.argmax(y_, axis = 1), predictions = predict_label,)[1]

	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

	# -------------------------------------- #
	sess = tf.Session()
	sess.run(init_op)

	# a class to save the variables of the network
	saver = tf.train.Saver()

	for i in range(1000):
		# load the training data
		x_train, y_train = mnist.train.next_batch(BATCH_SIZE)
		# feed the training data into graph ,train and compute loss
		_, loss_ = sess.run([train_op, loss], feed_dict = {x_: x_train, y_: y_train})

		if i % 50 == 0:
			accuracy_ = sess.run([accuracy], feed_dict = {x_: x_val, y_: y_val})
			print('step', i,': loss: %.4f' % loss_, 'validate accuracy: %.2f' % accuracy_[0])

	if save_network:
		saver.save(sess = sess, save_path = './NN', write_meta_graph = False)
		# meta_graph is not recommended

def reload():
	x_ = tf.placeholder(dtype = tf.float32, shape = [None, 28 * 28])
	y_ = tf.placeholder(dtype = tf.float32, shape = [None, 10])

	nn = tf.layers.dense(
		inputs = x_,
		units = Hidden,
		activation = tf.nn.relu,
		use_bias = True,
		kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01),
		bias_initializer = tf.constant_initializer(0.1),
		name = 'hidden_layer',
		reuse = None)

	output = tf.layers.dense(
		inputs = nn,
		units = 10,
		activation = tf.nn.softmax,
		kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01),
		bias_initializer = tf.constant_initializer(0.1),
		name = 'output_layer',
		reuse = None)

	predict_label = tf.argmax(output, axis = 1)

	loss = tf.losses.softmax_cross_entropy(onehot_labels = y_, logits = output)

	train_op = tf.train.AdamOptimizer(Learning_Rate).minimize(loss)

	accuracy = tf.metrics.accuracy(
		labels = tf.argmax(y_, axis = 1), predictions = predict_label,)[1]

	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	# --------------------------------------------------#

	sess = tf.Session()
	# must initialize the variables before reload the network
	sess.run(init_op)

	saver = tf.train.Saver()
	if load_network:
		saver.restore(sess = sess, save_path = './NN')

	# test
	x_test = mnist.test.images[:-test_len]
	y_test = mnist.test.labels[:-test_len]
	test_acc = sess.run([accuracy], feed_dict = {x_: x_test, y_:y_test})
	print('test accuracy: %.4f' % test_acc[0])

train()

# reset the graph
tf.reset_default_graph()

reload()