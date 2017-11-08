'''
Created on Jun 16, 2017

@author: xli22
'''
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from CNN import CNNmodel

from pyspark import SparkContext, SparkConf


def train(datasets):
		import tensorflow as tf
		sess = tf.Session()
		graph = CNNmodel()
		sess.run(tf.initialize_all_variables())
		iters = datasets.shape[0]//100
		for _ in range(2):
			for i in range(iters):
				batch_xs = datasets[i*100:(i+1)*100, 0:784]
				batch_ys = datasets[i*100:(i+1)*100, 784:]
				sess.run(graph.train_step, feed_dict={graph.xs:batch_xs, graph.ys:batch_ys})
			
		test_xs = test_bc.value[:,0:784]
		outlist = sess.run(graph.y_pred,feed_dict={graph.xs:test_xs})
		sess.close()
		
		return outlist

if __name__ == '__main__':
	sf = SparkConf().setAppName("SparkTF")
	sc = SparkContext(conf=sf)

	mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
	#trainset = np.hstack((mnist.train.images, mnist.train.labels))
	#np.savetxt("MNIST_data/train.txt", trainset, delimiter=',')
	testset = mnist.test.images[0:1000]
	testlabel = mnist.test.labels[0:1000]
	#np.savetxt("MNIST_data/test.txt", testset, delimiter=',')

	test_bc = sc.broadcast(testset)
					
	trainRDD = sc.textFile("MNIST_data/train.txt", 100)				\
				.map(lambda x: [float(i) for i in x.split(",")]) \
				.map(lambda x: (np.random.randint(0,100),x))

	trainBlock = trainRDD.groupByKey().map(lambda x: (x[0], np.asarray(list(x[1]))))

	pred = trainBlock.map(lambda x: train(x[1]))    \
					 .map(lambda x: [np.argmax(i) for i in x]).collect()
	pred = np.array(pred)

	accu = 0
	for i in range(len(testlabel)):
		if np.argmax(np.bincount(pred[:,i])) == np.argmax(testlabel[i]):
			accu = accu + 1
	
	accuracy = accu/len(testlabel)
	
	print(accuracy)
