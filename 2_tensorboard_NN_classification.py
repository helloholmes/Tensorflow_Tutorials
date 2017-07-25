# coding:utf-8
'''
python 3.5
tensorflow: 1.2.0
author: helloholmes

--------------------
after running this code
in terminal or CMD, type:
tensorboard --logdir path
open 'http://localhost:6006' in your browser
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

with tf.name_scope('network'):
	with tf.variable_scope('input'):
		x_ = tf.placeholder(dtype = tf.float32, shape = [None, 28 * 28], name = 'x_')
		y_ = tf.placeholder(dtype = tf.float32, shape = [None, 10], name = 'y_')

	with tf.variable_scope('hidden_layer'):
		nn = tf.layers.dense(
			inputs = x_,
			units = Hidden,
			activation = tf.nn.relu,
			use_bias = True,
			kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01),
			bias_initializer = tf.constant_initializer(0.1),
			name = 'hidden_layer',
			reuse = None)
		tf.summary.histogram(name = 'hidden_nn', values = nn)

	with tf.variable_scope('output'):
		output = tf.layers.dense(
			inputs = nn,
			units = 10,
			activation = tf.nn.softmax,
			kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01),
			bias_initializer = tf.constant_initializer(0.1),
			name = 'output_layer',
			reuse = None)
		tf.summary.histogram(name = 'prediction', values = output)

with tf.name_scope('loss'):
	loss = tf.losses.softmax_cross_entropy(onehot_labels = y_, logits = output)
	tf.summary.scalar('loss', loss)

with tf.name_scope('train_op'):
	train_op = tf.train.AdamOptimizer(Learning_Rate).minimize(loss)

with tf.name_scope('train_acc'):
	train_acc = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
	train_acc = tf.reduce_mean(tf.cast(train_acc, tf.float32))
	tf.summary.scalar('train_acc', train_acc)

with tf.name_scope('accuracy'):
	accuracy = tf.metrics.accuracy(
		labels = tf.argmax(y_, axis = 1), predictions = tf.argmax(output, axis = 1))[1]
	# tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('init_op'):
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# the graph has been created
# ------------------------------------------ #
sess = tf.Session()
sess.run(init_op)

writer = tf.summary.FileWriter('./log', sess.graph)
merge_op = tf.summary.merge_all()
# merge_acc = tf.summary.scalar('accuracy', accuracy)

# ------------------------------------------ #
# training started
for i in range(1000):
	# load the training data
	x_train, y_train = mnist.train.next_batch(BATCH_SIZE)
	# feed the training data into graph ,train and compute loss
	_, loss_, result = sess.run([train_op, loss, merge_op], feed_dict = {x_: x_train, y_: y_train})
	writer.add_summary(result, i + 1)

	if i % 50 == 0:
		accuracy_ = sess.run([accuracy], feed_dict = {x_: x_val, y_: y_val})
		print('step', i,': loss: %.4f' % loss_, 'validate accuracy: %.2f' % accuracy_[0])

# test
x_test = mnist.test.images[:-test_len]
y_test = mnist.test.labels[:-test_len]
test_acc = sess.run([accuracy], feed_dict = {x_: x_test, y_:y_test})
print('test accuracy: %.4f' % test_acc[0])

