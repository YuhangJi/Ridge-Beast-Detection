# Ridge-Beast-Detection
![image](https://github.com/YuhangJi/Ridge-Beast-Detection/blob/main/demo/demo.jpg)
## Introduction
  An implementation of improved nural network based on YOLOv3.  
## Core Proposals
  1.Deep Polymeric Features  
  2.SE block  
  3.Multi-scale Convolutional Structure  
  4.Open Source Dataset: [link](https://blog.csdn.net/weixin_45482843/article/details/106905824)
## Environment
  1.tensorflow 2.3  
  2.numpy  
  3.pillow  
  4.opencv  
  5.matplotlib
## How to use
### Data Preparing
  If you want to train your own dataset using this model, you need to label your dataset. And the annotation tool has been provided. You are advised to use it within english directory. If you want to train a model for detecting the ridge beast, although few people will like to do this, please download corresponding dataset as above link of open source dataset.  
### Data splitting
  The py file "./data_process/split_data.py" is to split dataset into train set and test set. You also use exsited "test.txt" and "train.txt" to get a result as the same as my works, but you have to write new code to handle the part of image directory by youself.  
### Configuration
  "./config.py" includes whole parameters about training and testing.
### Training
  Input the command "python main.py" to train a new model.
### Testing
  "./evaluation.py" is helping for getting a map on your dataset as well as you can change the "CONFIDENCE_THRESHOLD" and "IOU_THRESHOLD" to observe diffrent performance and the later one is 0.5 by default.