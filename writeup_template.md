# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[viz1]: ./examples/visualization1.jpg "Dataset Distribution"
[viz2]: ./examples/visualization2.jpg "Traffic Sign sample"
[viz3]: ./examples/visualization3.jpg "Image Augmentation"
[viz4]: ./examples/augmented-data-distribution.jpg "Dataset Distribution after augmentation"
[viz5]: ./examples/visualization5.jpg "Image Flipping"
[viz6]: ./examples/visualization6.jpg "Performance on new images"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/gts_1.jpg "Road work"
[image5]: ./examples/gts_2.jpg "Yield"
[image6]: ./examples/gts_3.jpg "Priority road"
[image7]: ./examples/gts_4.jpg "Stop"
[image8]: ./examples/gts_5.jpg "General Caution"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribution varies between training, validation and test dataset.

#### Remarks:
* As we can see, all the dataset is skewed. Some classes have more samples than others.

![alt text][viz1]

To further explore, the actual traffic sign samples are also visualized.

#### Remarks:
* The images vary greatly in brightness, rotation, background, warping etc.

![alt text][viz2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I have generated additional data based to provide more training samples and also combat data variance related to brightness, rotation etc.

To generate additional data, I mainly used techniques suggested in project rubrics.
* Flipping
* Rotation
* Scaling
* Histogram Equalization
* Intensity Rescaling
* Gamma Adjustment

This is an example of a traffic sign image before and after flipping method applied:
![alt text][viz5]

This is an example of a traffic sign image before and after augmentation method applied:
![alt text][viz3]

For traffic signs having less than 600 samples, new image per augmentation technique was added to the dataset. For other signs, new images were added randomly.

The distribution between the original data set and the augmented data set is as follows:
![alt text][viz4]

The distribution is mostly same, however the number of samples increased.

- Training samples before augmentation: 34799
- Training samples before augmentation: 103079

Before passing to training pipeline, the augmented dataset is converted to grayscale.

![alt text][image2]

Batch normalization step was added to the original LeNet architecture to reduce influence due to random weight initialization and to facilitate faster convergence.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5x1x6     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| Batch normalization | outputs 28x28x6 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5x1x16     	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		| outputs 400        									|
| Fully connected		| outputs 120        									|
| Fully connected		| outputs 84        									|
| Fully connected		| outputs 43        									|
| Softmax				| outputs 43        									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

|Model Property| Configuration |
|:------------:|:-------------:|
|Optimizer | Adam |
|Batch Size| 32 |
|Number of epochs| 20 |
|Learning rate| 0.001 |
|Weight initialization|tensorflow.truncated_normal(mu=0, sigma=0.1)|
|Batch normalization|epsilon=1e-5|

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.65%
* validation set accuracy of 96.87%
* test set accuracy of 93.60%

If a well known architecture was chosen:
* What architecture was chosen?
 - LeNet-5 architecture was chosen as suggested by Udacity team.
 - However, a batch normalization layer was added to skip normalization during preprocessing.

* Why did you believe it would be relevant to the traffic sign application?
 - As LeNet has considerable performance for MNIST image classification, I was convinced to use for traffic sign classification as well.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 - For all the dataset, final model accuracy is above the acceptable level of accuracy (93%). However, the model is overfitting as accuracy for validation dataset is higher than that of test dataset.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

| Image			        |     Comment	        					|
|:---------------------:|:---------------------------------------------:|
|![alt text][image8]|Moderately easy|
|![alt text][image5]|Moderately easy|
|![alt text][image4]|Difficult due to blur|
|![alt text][image7]|Moderately Difficult due to background noise|
|![alt text][image6]|Moderately easy|

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| General Caution			| General Caution      							|
| Yield					    | Yield											|
| Road Work	      		| Road Work					 				|
| Stop Sign      		| Stop sign   									|
| Priority Road     			| Priority Road	|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model is pretty certain about its predictions for each sign. However, for Stop sign, it is bit confused with Turn left ahead. Probably this is caused by training sample with background noise.

![alt text][viz6]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
