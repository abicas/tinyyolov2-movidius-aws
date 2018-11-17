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
* [Darknet] (https://pjreddie.com/darknet/)
* [Yolo (You Only Look Once)] (https://pjreddie.com/darknet/yolov2/)Object Detection 
* [Darkflow] (https://github.com/thtrieu/darkflow) to help transform Yolo model into Tensorflow

### Edge Device 
The idea behind this project is to allow for low cost devicdes to perform near real time inference. ALthough it would be possible to run the models entirely on low power CPU, if we need something near real time we will need to boost its processing capacity with an accelerator. 

* Raspberry Pi 3B+ 
* 16GB card 
* Raspbian 
* USB Webcam 
* [Movidius USB Stick] (https://www.movidius.com/) for accelerated

## Preparing the Training Environment 

In this section we will install and prepare the EC2 instance to run our training jobs. We will not go thru every step of the instance creation, assuming you already have an AWS account and knows how to deploy and properly configure an instance for SSH access. 



