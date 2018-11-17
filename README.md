# tinyyolov2-movidius-aws
Custom training Object Detection on Tiny-YoloV2 on AWS and then transfering the model to Movidius to perform accelerated inference on  edge devices

## Introduction

When you have Deep Learning models running on edge devices sending all the data to cloud services once these devices are not powerful enough to do the whole processing. Intel has created a small device called <a href="https://www.movidius.com/"> Movidius </a> that can act as an accelerator for edge devices like a Raspberry Pi.

Of course there are several limitations regarding the size of the model and specifically the supported operations, layers and networks that it can run. The <a href="https://movidius.github.io/ncsdk/release_notes.html"> Movidius SDK Release Notes </a> denotes some of these limitations.  

One network that I found to be really handy for object detection and that can be converted to run on a Raspberry Pi with Movidus is <a href="https://pjreddie.com/darknet/yolo/"> Tiny Yolo v2 </a> once it is fast to train, transfer learning does not create additional layers and it can pass thru the <a href="https://github.com/thtrieu/darkflow"> Darkflow  </a> transformations required to go from Yolo to Tensorflow and then to Movidius format. 

## Requirements

For this project there will be two environments: Training and Edge Ddevice

### Training Environment 
For this environment we will be using a server with an Nvidia GPU for training. 
I would recommend an AWS EC2 p3.2xlarge instance that contains a single V100 GPU making it simpler to train than multiple GPUs and still affordable enough for this project (as of writing around $3/hour). 

* AWS EC2 p3.2xlarge instance 
* AWS Deep Learning AMI (Deep Learning AMI (Ubuntu) Version 18.0 - ami-0484cefb8f48dafe8) 
* [Darknet](https://pjreddie.com/darknet/)
* [Yolo (You Only Look Once)](https://pjreddie.com/darknet/yolov2/)Object Detection 
* [Darkflow](https://github.com/thtrieu/darkflow) to help transform Yolo model into Tensorflow

### Edge Device 
The idea behind this project is to allow for low cost devicdes to perform near real time inference. ALthough it would be possible to run the models entirely on low power CPU, if we need something near real time we will need to boost its processing capacity with an accelerator. 

* Raspberry Pi 3B+ 
* 16GB card 
* Raspbian 
* USB Webcam 
* [Movidius USB Stick](https://www.movidius.com/) for accelerated

## Setting things up for Training

### Preparing the Training Instance 

In this section we will install and prepare the EC2 instance to run our training jobs. We will not go thru every step of the instance creation, assuming you already have an AWS account and knows how to deploy and properly configure an instance for SSH access. The main steps you have to watch for while creating the server are described below: 

We will use the *Deep Learning AMI (Ubuntu) Version 18.0 - ami-0484cefb8f48dafe8* to create our instance (there can be a new version of the AMI when you deploy your instance once they get updated quite frequently, but it shouldn't be an issue for the project in case you decide to use a newwer one):
![EC2 AMI](images/ami1.png)

In order to choose an instance with GPU support, choose p3.2xlarge as the instance type (as the time of writing this instance will cost you around $3 per active hour plus the storage space):
![EC2 instance](images/ami2.png)

Also, if you need to access it thru SSH externally you can assign a Public IP to it
![EC2 Public IP](images/ami3.png)

Accepting the other defaults should be alright for the project. 

Thanks to the AMI, the Instance comes up preinstalled with many Deep Learning tools already set up, as well as some prerequisites like CUDA and Python Libraries, making our life way much easier. 

SSH the newly created instance and activate the *python2* environment with `source activate python2`
![source activate python2](images/ami4.png)

### Installing Darknet

Let's move to the next step, installing the required software to training the models. You can follow the guide directly from [Darknet website](https://pjreddie.com/darknet/install/) or cut directly to the steps here: 
````
git clone https://github.com/pjreddie/darknet.git
cd darknet
````
In order to have CUDA enabled we need to change the ````Makefile```` changing the following parameters from 0 to 1 
````
GPU=1
CUDNN=1
OPENMP=1
````
Then compile it runing ````make````
If everything goes fine, you have an executable in your working dir. Let's test it ! 
![test run](images/darknet1.png)
The message shows it has compiled correctly. 




