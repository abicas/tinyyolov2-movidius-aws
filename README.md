# tinyyolov2-movidius-aws
Training Object Detection on Tiny-YoloV2 on AWS and then transfering the model to Movidius to perform accelerated inference on the edge

## Introduction

When you have Deep Learning models running on edge devices sending all the data to cloud services once these devices are not powerful enough to do the whole processing. Intel has created a small device called <a href="https://www.movidius.com/"> Movidius </a> that can act as an accelerator for edge devices like a Raspberry Pi.

Of course there are several limitations regarding the size of the model and specifically the supported operations, layers and networks that it can run. The <a href="https://movidius.github.io/ncsdk/release_notes.html"> Movidius SDK Release Notes </a> denotes some of these limitations.  

One network that I found to be really handy for object detection and that can be converted to run on a Raspberry Pi with Movidus is <a href="https://pjreddie.com/darknet/yolo/"> Tiny Yolo v2 </a> once it is fast to train, transfer learning does not create additional layers and it can pass thru the <a href="https://github.com/thtrieu/darkflow> Darkflow  </a> transformations required to go from Yolo to Tensorflow and then to Movidius format. 
