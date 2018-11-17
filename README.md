# Tiny Yolo V2 ported to run on Edge Devices powered by Movidius, running on AWS 
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

In this section we will install and prepare the EC2 instance to run our training jobs. We will not go thru every step of the instance creation, assuming you already have an AWS account and knows how to deploy and properly configure an instance for SSH access. 

We have provided a [Cloudformation script](create_training_instance_us-east-1.yaml) to create the instance but if you want do customized it or deploy in another region than US-EAST-1, the main steps you have to watch for while creating the server are described below: 

We will use the *Deep Learning AMI (Ubuntu) Version 18.0 - ami-0484cefb8f48dafe8* to create our instance (there can be a new version of the AMI when you deploy your instance once they get updated quite frequently, but it shouldn't be an issue for the project in case you decide to use a newwer one):
![EC2 AMI](images/ami1.png)

In order to choose an instance with GPU support, choose p3.2xlarge as the instance type (as the time of writing this instance will cost you around $3 per active hour plus the storage space):
![EC2 instance](images/ami2.png)

Also, if you need to access it thru SSH externally you can assign a Public IP to it
![EC2 Public IP](images/ami3.png)

Accepting the other defaults should be alright for the project. 

Thanks to the AMI, the Instance comes up preinstalled with many Deep Learning tools already set up, as well as some prerequisites like CUDA and Python Libraries, making our life way much easier. 

![source activate python2](images/ami4.png)
SSH the newly created instance and activate the *python2* environment with `source activate python2`

### Installing Darknet

Let's move to the next step, installing the required software to training the models. You can follow the guide directly from [Darknet website](https://pjreddie.com/darknet/install/) or cut directly to the steps here: 
````bash
git clone https://github.com/pjreddie/darknet.git
cd darknet
````
In order to have CUDA enabled we need to change the ````Makefile```` changing the following parameters from 0 to 1 
````
GPU=1
CUDNN=1
OPENMP=1
````
Then compile it runing ````make````.  

If everything goes fine, you have an executable in your working dir. Let's test it ! 
![test run](images/darknet1.png)  
The message shows it has compiled correctly. 

### Testing Darknet with Tiny YOLO v2

In order to run our test, we need to download a pre-trained weights file. There are several files available. We will be using Tiny Yolo v2 due to its speed and size, small enough to be converted to Movidius and loaded into a raspberry pi memory later on (although less accurate). 

Let's download a pre-trained model of TinyYoloV2 trained on VOC data and run it agains a dog picture

````bash
wget https://pjreddie.com/media/files/yolov2-tiny-voc.weights

./darknet detector test cfg/voc.data cfg/yolov2-tiny-voc.cfg yolov2-tiny-voc.weights data/dog.jpg

layer     filters    size              input                output
    0 conv     16  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  16  0.150 BFLOPs
    1 max          2 x 2 / 2   416 x 416 x  16   ->   208 x 208 x  16
    2 conv     32  3 x 3 / 1   208 x 208 x  16   ->   208 x 208 x  32  0.399 BFLOPs
    3 max          2 x 2 / 2   208 x 208 x  32   ->   104 x 104 x  32
    4 conv     64  3 x 3 / 1   104 x 104 x  32   ->   104 x 104 x  64  0.399 BFLOPs
    5 max          2 x 2 / 2   104 x 104 x  64   ->    52 x  52 x  64
    6 conv    128  3 x 3 / 1    52 x  52 x  64   ->    52 x  52 x 128  0.399 BFLOPs
    7 max          2 x 2 / 2    52 x  52 x 128   ->    26 x  26 x 128
    8 conv    256  3 x 3 / 1    26 x  26 x 128   ->    26 x  26 x 256  0.399 BFLOPs
    9 max          2 x 2 / 2    26 x  26 x 256   ->    13 x  13 x 256
   10 conv    512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 x 512  0.399 BFLOPs
   11 max          2 x 2 / 1    13 x  13 x 512   ->    13 x  13 x 512
   12 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   13 conv   1024  3 x 3 / 1    13 x  13 x1024   ->    13 x  13 x1024  3.190 BFLOPs
   14 conv    125  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 125  0.043 BFLOPs
   15 detection
mask_scale: Using default '1.000000'
Loading weights from yolov2-tiny-voc.weights...Done!
data/dog.jpg: Predicted in 0.002161 seconds.
dog: 78%
car: 55%
car: 51%
````
![dog](https://github.com/pjreddie/darknet/blob/master/data/dog.jpg)   
Neat right ? It was able to spot the dog if 78% certainty in 0.002 seconds (and the car in the background as well)! But this is running on a big server, with a NVIDIA V100 GPU, running a pretty known and optimized model... there will be some hard work ahead in order to customize this to our data and run it on less powerfull devices. But we will make it ! 

### Installing Darkflow 

Movidius devices can only compile deep learning models in Caffe or Tensorflow. So we need to convert the darknet to Tensorflow so it can be converted later on to Movidius. This convertion to Tensorflow will be done my [Darkflow](https://github.com/thtrieu/darkflow). 

But why don't you do the training already on Tensroflow them? Well, based on my experiences, Movidius supports quite a limited subset of operators of Tensorflow and doing transfer learning on existing models to my classes wasn't generating models capable of being converted. The [release notes from Movidius](https://movidius.github.io/ncsdk/release_notes.html) are always changing, so I would recommend checking back to see if it is supporting your required operators. DArknet on the other hand is a network that can be converted without incurring in these issues through the use of Darkflow. 

Darkflow runs on Python3, so lets switch environments and download the code: 
`````bash
cd ~
source deactivate
source activate python3
git clone https://github.com/thtrieu/darkflow.git
`````
Now let's move to the pre-requisites and Cython extension install: 
````bash 
cd ~/darkflow
pip install tensorflow
pip install opencv-python
pip install -e .
````
Lets test darkflow install. 
First lets use our existing darkflow model and run it thru the pictures on `sample_img` dir, using the pre-trained models we downloaded previously:
````bash
## Run the model and generate an output image in ~/darkflow/sample_img/out/
python3 flow --imgdir sample_img/ --model ../darknet/cfg/yolov2-tiny-voc.cfg --load ../darknet/yolov2-tiny-voc.weights  --labels ../darknet/data/voc.names

## Run the model and generate a JSON with the output results in ~/darkflow/sample_img/out/
python3 flow --imgdir sample_img/ --model ../darknet/cfg/yolov2-tiny-voc.cfg --load ../darknet/yolov2-tiny-voc.weights  --json --labels ../darknet/data/voc.names

cat sample_img/out/sample_dog.json

[{"label": "car", "confidence": 0.77, "topleft": {"x": 444, "y": 90}, "bottomright": {"x": 685, "y": 186}}, {"label": "dog", "confidence": 0.8, "topleft": {"x": 96, "y": 222}, "bottomright": {"x": 344, "y": 532}}]
````
![dogdarflow](images/sample_dog.jpg)



Now lets test a conversion from Darknet to Tensorflow: 
````bash
## Convert the Darknet Tiny-Yolo v2 Model to Tensorflow and save it in built_graph/

python3 flow --savepb --model ../darknet/cfg/yolov2-tiny-voc.cfg --load ../darknet/yolov2-tiny-voc.weights  --labels ../darknet/data/voc.names
Parsing ../darknet/cfg/yolov2-tiny-voc.cfg
Loading ../darknet/yolov2-tiny-voc.weights ...
Successfully identified 63471556 bytes
Finished in 0.00496363639831543s

Building net ...
Source | Train? | Layer description                | Output size
-------+--------+----------------------------------+---------------
       |        | input                            | (?, 416, 416, 3)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 416, 416, 16)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 208, 208, 16)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 208, 208, 32)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 104, 104, 32)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 104, 104, 64)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 52, 52, 64)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 52, 52, 128)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 26, 26, 128)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 26, 26, 256)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 13, 13, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 512)
 Load  |  Yep!  | maxp 2x2p0_1                     | (?, 13, 13, 512)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 1x1p0_1    linear           | (?, 13, 13, 125)
-------+--------+----------------------------------+---------------
Running entirely on CPU
2018-11-17 20:27:45.415208: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Finished in 1.7518539428710938s

Rebuild a constant version ...
Done

ls built_graph/
yolov2-tiny-voc.meta  yolov2-tiny-voc.pb
````
Let's test the newly converted model 
````bash
## Run the model and generate an output image in ~/darkflow/sample_img/out/
python3 flow --pbLoad built_graph/yolov2-tiny-voc.pb --metaLoad built_graph/yolov2-tiny-voc.meta --imgdir sample_img/


## Run the model and generate a JSON with the output results in ~/darkflow/sample_img/out/
python3 flow --pbLoad built_graph/yolov2-tiny-voc.pb --metaLoad built_graph/yolov2-tiny-voc.meta --imgdir sample_img/ --json

Loading from .pb and .meta
WARNING:tensorflow:From /home/ubuntu/darkflow/darkflow/net/build.py:81: FastGFile.__init__ (from tensorflow.python.platform.gfile) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.gfile.GFile.
Running entirely on CPU
2018-11-17 20:32:43.767603: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Forwarding 8 inputs ...
Total time = 1.3310556411743164s / 8 inps = 6.010267153777317 ips
Post processing 8 inputs ...
Total time = 0.27250194549560547s / 8 inps = 29.357588568588817 ips 

cat sample_img/out/sample_dog.json
[{"label": "car", "confidence": 0.77, "topleft": {"x": 444, "y": 90}, "bottomright": {"x": 685, "y": 186}}, {"label": "dog", "confidence": 0.8, "topleft": {"x": 96, "y": 222}, "bottomright": {"x": 344, "y": 532}}]
````
![dogdarflow](images/sample_dog2.jpg)

Before continuing we need to do a minor change on the source of a file in Darkflow. It was designed for pre-trained weights of Yolo, but once we do a custom training it adds 4 bytes to the begining of its weight file that needs to be padded on the reading. 

Edit the file `~/darkflow/darkflow/utils/loader.py` and change `self.offset = 16` to `self.offset = 20` in `weights_walker.__init__()` (line 121)

## Custom Training Tiny Yolo v2 

For this project we will train the network to recognize Logos of 27 brands

### Downloading and Preparing training data 

In order to train the network we will use a Dataset gathered from Flickr named [Flickr Logos Dataset](http://image.ntua.gr/iva/datasets/flickr_logos/).  It contains 810 annotated images, corresponding to 27 logo classes/brands (30 images for each class). All images are annotated with bounding boxes of the logo instances in the image.

So, let's move on and gather this data on a separate directory once it will need to be prepared before training: 
````bash
source deactivate
source activte python2
mkdir ~/flickr27
cd ~/flickr27
wget http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz
gunzip flickr_logos_27_dataset.tar.gz
tar -xvf flickr_logos_27_dataset.tar
cd flickr_logos_27_dataset/
gunzip flickr_logos_27_dataset_images.tar.gz
tar -xvf flickr_logos_27_dataset_images.tar.gz

ls -la
total 102592
drwxrwxr-x 3 ubuntu ubuntu      4096 Nov 17 21:07 .
drwxrwxr-x 3 ubuntu ubuntu      4096 Nov 17 21:07 ..
-rw-rw-r-- 1 ubuntu ubuntu    264933 Apr 26  2011 flickr_logos_27_dataset_distractor_set_urls.txt
drwxrwxr-x 2 ubuntu ubuntu     36864 Apr 26  2011 flickr_logos_27_dataset_images
-rw-rw-r-- 1 ubuntu ubuntu 104550400 Apr 26  2011 flickr_logos_27_dataset_images.tar
-rw-rw-r-- 1 ubuntu ubuntu      5850 Apr 26  2011 flickr_logos_27_dataset_query_set_annotation.txt
-rw-rw-r-- 1 ubuntu ubuntu    182683 Apr 26  2011 flickr_logos_27_dataset_training_set_annotation.txt

head -n 10 flickr_logos_27_dataset_training_set_annotation.txt
144503924.jpg Adidas 1 38 12 234 142
2451569770.jpg Adidas 1 242 208 413 331
390321909.jpg Adidas 1 13 5 89 60
4761260517.jpg Adidas 1 43 122 358 354
4763210295.jpg Adidas 1 83 63 130 93
4763210295.jpg Adidas 1 91 288 125 306
4763210295.jpg Adidas 1 182 63 229 94
4763210295.jpg Adidas 1 192 291 225 306
4763210295.jpg Adidas 1 285 61 317 79
4763210295.jpg Adidas 1 285 298 324 329
````
You will notice we have a directory with all the images and 3 txt files. 
The file `flickr_logos_27_dataset_training_set_annotation.txt`  contains the main info we need for training: Picture, Class, Subset, Top, Left, Bottom, Right in pixels for the objects. A file can have more than on object like `4763210295.jpg` in the example above with many lines poiting to the same file. 

Darknet uses a different file format for training where each picture `.jpg` has a file with the `.txt` extension containing in each line the ID for the class (integer) plus the object coordinates relative to the file size (top, left, width, height). 

Instead of manually changing the format, you can use the provided script `convert_Flickr2Yolo.py` to convert the labels into Yolo format: 
````bash
cd ~/flickr27/flickr_logos_27_dataset/
mkdir tmp
python convert_Flickr2Yolo.py
flickr_logos_27_dataset_images/144503924.jpg--->tmp/1_Adidas_144503924.jpg(tmp/1_Adidas_144503924.txt)
flickr_logos_27_dataset_images/2451569770.jpg--->tmp/1_Adidas_2451569770.jpg(tmp/1_Adidas_2451569770.txt)
flickr_logos_27_dataset_images/390321909.jpg--->tmp/1_Adidas_390321909.jpg(tmp/1_Adidas_390321909.txt)
flickr_logos_27_dataset_images/4761260517.jpg--->tmp/1_Adidas_4761260517.jpg(tmp/1_Adidas_4761260517.txt)
.
.
.
flickr_logos_27_dataset_images/217288720.jpg--->tmp/6_Yahoo_217288720.jpg(tmp/6_Yahoo_217288720.txt)
flickr_logos_27_dataset_images/2472817996.jpg--->tmp/6_Yahoo_2472817996.jpg(tmp/6_Yahoo_2472817996.txt)
flickr_logos_27_dataset_images/2514220918.jpg--->tmp/6_Yahoo_2514220918.jpg(tmp/6_Yahoo_2514220918.txt)
flickr_logos_27_dataset_images/386891249.jpg--->tmp/6_Yahoo_386891249.jpg(tmp/6_Yahoo_386891249.txt)
=========== DONE
CATEGORIES LIST for LABELS
['Pepsi', 'Puma', 'Ferrari', 'Sprite', 'Ford', 'HP', 'Fedex', 'Starbucks', 'DHL', 'Google', 'Heineken', 'RedBull', 'Intel', 'Nike', 'Porsche', 'Adidas', 'McDonalds', 'Citroen', 'Texaco', 'Unicef', 'Yahoo', 'BMW', 'Nbc', 'Cocacola', 'Vodafone', 'Apple', 'Mini']
`````
The script has created the required files in `tmp/` directory. It also provides a `labels.txt` on the base directory that will be used in training and a *CATEGORIES LIST for LABELS* that will be required later on for the Inference, so take note of these values. 

Next we need to copy the files and generate the required structure for training:
````bash
cd ~/darknet
mkdir ~/darknet/data/logos/
cp ~/flickr27/flickr_logos_27_dataset/tmp/*  ~/darknet/data/logos/
ls ./data/logos/*jpg | grep -v 6_ > ~/darknet/data/train.txt
ls ./data/logos/*jpg | grep  6_ > ~/darknet/data/test.txt
cp ~/flickr27/flickr_logos_27_dataset/logos.txt ~/darknet/data/logos.txt
cp cfg/voc.data cfg/logos.data
cp cfg/yolov2-tiny-voc.cfg cfg/yolov2-tiny-logos.cfg
````
Edit the file `cfg/logos.data` to: 
````bash
classes= 27
train  = ./data/train.txt
valid  = ./data/test.txt
names = ./data/logos.txt
backup = backup
````
Ok ! So now we have the annotations in the correct format, the labels in the right place, a list of images for training (1_*.jpg thru 5_*.jpg), a list of images for validation (6_*.jpg), and a file containing all the paths we need. 

The default `.cfg` file is configured for 20 classes and testing. We need to change it to more classes and training. 

Open the `cfg/yolov2-tiny-logos.cfg`in your edit and change the following lines to:

````bash
1 [net]
2 # Testing
3 #batch=1
4 #subdivisions=1
5 # Training
6 batch=128
7 subdivisions=8
 .
 .
 .
114 [convolutional]
115 size=1
116 stride=1
117 pad=1
118 filters=160
119 activation=linear
 . 
 . 
 .
124 classes=27
125 coords=4
126 num=5
````
We commented lines 3 and 4 and changed lines 6 and 7 to allow training in larger batches suitable for the larger GPU in P3 instances. We also changed classes in line 124 to 27 and filters on line 118. Filters must be changed to the value defined by "(classes + 5) * 5"... in this case (27+5)*5=160. 


















