# GNN_GPU
HPCC's GNN with GPU acceleration


## Description
Large nueral networks typically train on very large datasets. 
The size of the data combined with the size and complexity of neural networks result in large computational requirements. 
Until now, HPCC Systems is primarily a CPU based system which can result in neural network training times that are impractcally long.
With the use of modern GPUs, the training time can be drastically reduced over using CPUs alone.

This repository consists of a modified GNN bundle and example code showing how it is possible to train large neural networks on multiple GPUs using HPCC Systems. One can train a single NN model accross multiple GPUs that span 
multiple physical computers or even onto just one. Choosing how many GPUs to use is up to the developer as each model and dataset are different. Training time performance metrics are provided that demonstrate
the scalability of the setup with respect to training dataset size and neural network size (measured as a number of trainable feaetures).

## Getting Started

### Requirements
You must have a compatible NVIDIA GPU for the GPU acceleration to work, however this work consists of performance improvements-for the standard GNN-that can speed up the CPU training as well. The 
scope of this repo is limited to the use of GNN and GPUs though.


There is a AWS AMI that was created that you can use (ami-077ae915aa4f576fe), or you can generate your own with [this](https://github.com/xwang2713/cloud-image-build). 
This produces an image with HPCC Systems Platform Community edition, version 7.10.00 pre-installed as well as all other requirements for this bundle, including CUDA version 10.0. 
The image is designed to run on Amazon's [P2](https://aws.amazon.com/ec2/instance-types/p2/) or [P3](https://aws.amazon.com/ec2/instance-types/p3/) machines.


If you want to create your own instances from scratch, all you need is the latest HPCC systems installed on all nodes and these python packages.


Installation of packages used in the experiments:
```
server install commands:
Downloaded HPCC 7.10.0 files into working directory as *.deb

sudo apt-get update -y
sudo dpkg -i /tmp/*.deb
sudo apt-get install -f 

sudo apt-get update -y
yes | sudo apt-get install python3-pip
yes | sudo -H -u hpcc pip3 install --user pandas h5py


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update -y
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get install -y ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends nvidia-driver-450
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends cuda-10-0 libcudnn7=7.6.0.64-1+cuda10.0  libcudnn7-dev=7.6.0.64-1+cuda10.0

sudo -H -u hpcc pip3 install --user tensorflow-gpu==1.14 keras
```


### Included Training Data
Included is the MNIST [Dataset](Datasets/data_files) (see [reference](http://yann.lecun.com/exdb/mnist/)) used in in the experiments and exmaples. Additionally, the script to make
the two larger MNIST datasets (to simulate larger data) is included as a Jupyter notebook. 


#### Spraying
Spray in the following way, with appropriate names.

* MNIST: Fixed size = 785
	* Train: mnist::train
	* Test: mnist::test
* Medium MNIST: Fixed size = 785
	* Train: mnist::med::train
	* Test: mnist::med::test
* Large MNIST: Fixed size = 785
	* Train: mnist::big::train
	* Test: mnist::big::test

### Examples
Included in this bundle are some [examples](examples/), found in the examples directory. It is recomended to start with an [MLP trained on MNIST](examples/mnist_mlp.ecl)


The [examples](examples/) include how to properly load the image data from raw pixel values into something GNN can use to train NN models. It uses the above MNIST datasets 
to train three arbitrarily large CNN models on each of the three arbitrarily large image data. The code that was used to generate the experimental results are also provided.




## Author
Robert K.L. Kennedy | Summer 2020 | [GitHub](https://github.com/robertken) | [LinkedIn](https://www.linkedin.com/in/robertken/)


