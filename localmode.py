'''
Created on Jun 22, 2017

@author: xli22
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from CNN import CNNmodel

def train(datasets, testset):
		import tensorflow as tf
		session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
		sess = tf.Session(config=session_conf)
		graph = CNNmodel()
		sess.run(tf.initialize_all_variables())
		iters = datasets.shape[0]//100
		for _ in range(1):
			for i in range(iters):
				batch_xs = datasets[i*100:(i+1)*100, 0:784]
				batch_ys = datasets[i*100:(i+1)*100, 784:]
				sess.run(graph.train_step, feed_dict={graph.xs:batch_xs, graph.ys:batch_ys})
			
		test_xs = testset[:,0:784]	
		outlist = sess.run(graph.y_pred,feed_dict={graph.xs:test_xs})
		sess.close()
		
		return outlist
	
if __name__ == '__main__':
	mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
	#trainset = np.hstack((mnist.train.images, mnist.train.labels))
	#np.savetxt("MNIST_data/train.txt", trainset, delimiter=',')
	#trainset = trainset[0:1000]
	
	trainset = np.loadtxt("MNIST_data/train_data/train00.txt", delimiter=',') 
	
	testset = mnist.test.images[0:1000]
	testlabel = mnist.test.labels[0:1000]
	#np.savetxt("MNIST_data/test.txt", testset, delimiter=',')

	pred = train(trainset, testset)
	#pred = tf.equal(tf.argmax(pred,1), tf.argmax(testlabel,1))
	#acc = tf.reduce_mean(tf.cast(pred,tf.float32))

	accu = 0
	for i in range(len(testlabel)):
		if np.argmax(pred[i]) == np.argmax(testlabel[i]):
			accu = accu + 1
	
	accuracy = accu/len(testlabel)
	
	print(accuracy)
