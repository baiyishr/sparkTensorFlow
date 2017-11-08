# sparkTensorFlow
distributed tensorflow models on Spark

Tensorflow is a popular deep learning library for GPU and CPUs but the current version is not scalable. Here I proposed a simple design to train multiple tensorflow CNN models on a Spark cluster with random data samples from MNIST datasets. The final prediction was made by taking the majority vote of all the parallel models. In the early stage testing, this ensemble model improved the accuracy of predictions and reduced the running time of the whole training process, comparing to a single-core model.
