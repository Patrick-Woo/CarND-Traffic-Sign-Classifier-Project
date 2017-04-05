# **Traffic Sign Recognition** 

## Patrick Wu 2017/04/05

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/after_brightness.jpg "after_brightness"
[image3]: ./examples/after_rotation.jpg "after_rotation"
[image4]: ./examples/after_affine.jpg "after_affine"
[image5]: ./examples/normalization.JPG "normalization"
[image6]: ./examples/visualization_add.jpg "visualization with final train data"
[image7]: ./examples/20.jpg "limit 20km/h"
[image8]: ./examples/20_aug.jpg "limit 20km/h augmentation"
[image9]: ./genman_traffic_signs_unseen/test1.jpg 
[image10]: ./genman_traffic_signs_unseen/test2.jpg 
[image11]: ./genman_traffic_signs_unseen/test3.jpg 
[image12]: ./genman_traffic_signs_unseen/test4.jpg 
[image13]: ./genman_traffic_signs_unseen/test5.jpg 
[image14]: ./genman_traffic_signs_unseen/test6.jpg 
[image15]: ./genman_traffic_signs_unseen/test7.jpg 
[image16]: ./genman_traffic_signs_unseen/test8.jpg 
[image17]: ./examples/unseen_1.jpg 
[image18]: ./examples/unseen_2.jpg 
[image19]: ./examples/feature_map.jpg 
[image20]: ./examples/feature_map_1.jpg 
[image21]: ./examples/feature_map_2.jpg 
[image22]: ./examples/feature_map_3.jpg 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier-Lenet-Modified.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the following code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is three bar charts showing how the train/valid/test data distribute. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the transform_image function code cell of the IPython notebook.

As a first step, I decided to convert the images to different brightness because the weather condition can be different, for example from cloudy to sunny.


As a second step, I use the translation, rotation and affine transformation methods to preprocess the images to generalize the training data.

Here is an example of a traffic sign image before and after preprocessing.

![alt text][image2]
![alt text][image3]
![alt text][image4]

As a last step, I normalized the image data because this can make the gradient descent process faster.
![alt text][image5]



#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

As the data set had three parts, which are train, validation, test set and their distributions are almost the same, I did not preform further work to split them. However, I really did generating additional data for the training set to get better performance in the following training CNN process.

My final training set had 102179 number of images. My validation set and test set had 4410 and 12630 number of images.

I did the augmentation in the preprocess section which is in the 51st code cell of the notebook.


The difference between the original data set and the augmented data set is the following ... 

I would like to concentrate on discribing how I created addition data for the train set.

The 44th code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data for those classes whose examples are less than 1500.

And my final train set is consisted of three parts, the orinal train set(34799), one augmentaion image per image from train set(34799) and additional images(32581) from the classes whose examples are less than 1500.

So the final train set has 102179 number of images. I generated more data while keeping the distribution same as the original train set. This makes the prediction performance on valid/test set better as they all have the similar data distribution.

![alt text][image6]

To add more data to the the data set, I used the following techniques.

* brightness transform
* translation
* affine transform
* rotation

Here is an example of the original limit 20km/h image and 100 augmented additional fake images:

![alt text][image7]

![alt text][image8]


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 75th cell of the ipython notebook. 

My final model consists of 3 convolutional layers and 3 fully connected layers.

I use the techques of batch normalization and drop out to prevent overfitting.

Also batch normalization is very effective for preventing gradient vanishment.

I do not use L2 regulization as drop out and batch normalization are good enough to tackle the overfitting preblom.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x16 	|
| Batch Normalization					|												|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x16 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 16x16x32								|
| Batch Normalization					|												|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x32 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 8x8x64								|
| Batch Normalization					|												|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x64 				|
| Flatten					|outputs 1024												|
| Fully connected		| inputs 1024, outputs 512        									|
| Drop out					| keep prob 0.5												|
| Fully connected		| inputs 512, outputs 360        									|
| Drop out					| keep prob 0.5												|
| Fully connected		| inputs 360, outputs 43        									|
| Softmax				|         									|
|						|												|
|						|												|
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 128th cell of the ipython notebook. 

To train the model, I used hyper parameters of 
batch size=512, (I also test the batch size of 256, however it gives me more noise. so I increase the batch size.)

max epochs=100,

early stop without any improvement after 10 epoches , (10 epoches seem best as I also test 5,8 epoches.)

drop out percentage=0.5, (I use 0.6,0.7 for keep prob and find the default 0.5 is the best as we have enough data say 100,000+ data and I can drop out more percentage of data to handle overfitting problem.)

ExponentialMovingAverage decay rate=0.5

optimizer = Adam

learning rate=1e-3




#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 141st cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.959
* test set accuracy of 0.949

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I tried architectures like vgg 16 and Inception version 4 because they are notable and have reached a high performance in ImageNet.

* What were some problems with the initial architecture?
They are too deep and need a long time to train the whole model.I only has 43 classes not 1001 classes. Maybe we do not need so complicated model to address our traffic signs classification. Ultimately, I gave up and chose this modifed lenet model.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I used the modified lenet structure with 3 conv layers and 3 fc layers as they are easy enough to classify the traffic signs.
In terms of the hyper parameters tuning, please refer to the hyper parameter section above.

* Which parameters were tuned? How were they adjusted and why?
In terms of the hyper parameters tuning, please refer to the hyper parameter section above.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Batch norm ont only prevents gradient vanishment, but also tackle the problem of overfitting. Also, it can speed up the training process as data from every layer has the same scale.
Drop out use the bagging method to tackle the overfitting, while early stop saves the traing time as well as prevent overfitting.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12]  
![alt text][image13] ![alt text][image14] ![alt text][image15] ![alt text][image16]

The limit 30km/h image might be difficult to classify because it has other objects like a car and the number of 30 is small.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 145th cell of the Ipython notebook.

Here are the results of the prediction:
![alt text][image17]
![alt text][image18]



The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75%. This compares favorably to the accuracy on the test set of 94.9%. I think this model has a hard time classifing the images which contain many other objects like trees and cars as this kind of image is slightly different from the training set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 137th and 145th cells of the Ipython notebook.

For the most image, the model is relatively sure what the image is. However, for the last image which contains a limit 30 km/h traffic sign and other objects like a car, a road and trees, the model is really struggling to judge what the class is.

Also, this model is hard to classify the 30 and 50 km/h sign as 3 and 5 are similar in terms of their shapes.
![alt text][image18]

#### 4.VISUALIZE LAYERS OF THE NEURAL NETWORK
The code for  VISUALIZing LAYERS OF THE NEURAL NETWORK is located in the 148th and 152th cells of the Ipython notebook.

The source image is as follows:
![alt text][image19]

When looking around the feature maps from conv1 activation layer, I find they concains rough textures of the images.
![alt text][image20]

When looking around the feature maps from conv2 activation layer, I find they concains more detailed textures of the images.
![alt text][image21]

When looking around the feature maps from conv2 activation layer, I find they concains the most detailed textures of the images. As it is too detailed, I guess I should remove it and see whether the accuracy on test set will rise.
![alt text][image22]