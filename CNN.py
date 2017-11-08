'''
Created on Jun 16, 2017

@author: xli22
'''
import tensorflow as tf
import numpy as np

class CNNmodel:
	'''
	This class will build a CNN model
	'''
	#prediction=None
	##sess = None
	#xs = None
	#ys = None
	
	
	def __init__(self, input_shape=[None,784], num_classes=10):
		'''
		Constructor
		create a CNN model
		'''
		self.xs = tf.placeholder(tf.float32, shape=input_shape)
		self.ys = tf.placeholder(tf.float32, shape = [None, num_classes])

		x_image = tf.reshape(self.xs, [-1, 28, 28, 1])

		## conv1 layer
		W_conv1 = self.weight_variable([5,5,1,32])
		b_conv1 = self.bias_variable([32])
		h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1) 	#28x28x32
		h_pool1 = self.max_pool(h_conv1)									#14x14x32

		## conv2 layer
		W_conv2 = self.weight_variable([5,5,32,64])
		b_conv2 = self.bias_variable([64])
		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2) 	#14x14x64
		h_pool2 = self.max_pool(h_conv2)									#7x7x64

		## fc1 layer1
		W_fc1 = self.weight_variable([7*7*64,1024])
		b_fc1 = self.bias_variable([1024])
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) #1024
		h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

		## fc2 layer1
		W_fc2 = self.weight_variable([1024, 10])
		b_fc2 = self.bias_variable([10])
		self.y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)	
		
		##cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ys, logits=self.y_pred))
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cost)

		#pred = tf.equal(tf.argmax(self.y_pred,1), tf.argmax(self.ys,1))
		#acc = tf.reduce_mean(tf.cast(pred,tf.float32))

	def weight_variable(self, shape):
		return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

	def bias_variable(self, shape):
		return tf.Variable(tf.constant(0.1,shape=shape))

	def conv2d(self, x,W):
		return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

	def max_pool(self, x):
		return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')




		
		
		
		