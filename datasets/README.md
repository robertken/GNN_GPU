# Prepare MNIST Dataset
Download the MNIST dataset into a directory from [here](http://yann.lecun.com/exdb/mnist/)

* train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
* train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
* t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
* t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes))

Then use this [Script](prepare_mnist_data.ipynb) to prepare the MNIST data so it can be sprayed onto the HPCC cluster. You can also create arbitrarily large MNIST datasets for emulating larger datesets. You can spray the MNIST onto a cluster as such:
* MNIST: Fixed size = 785
	* Train: mnist::train
	* Test: mnist::test
	* Contains 60k training images
	* ~ 50 MB in size
* Medium MNIST: Fixed size = 785
	* Train: mnist::med::train
	* Test: mnist::med::test
	* Contains 600k training images
	* ~500 MB in size
* Large MNIST: Fixed size = 785
	* Train: mnist::big::train
	* Test: mnist::big::test
	* Contains 6M training images
	* ~ 5GB in size