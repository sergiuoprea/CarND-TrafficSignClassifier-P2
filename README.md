## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./figures/training_distr.png "Class distribution in the training set showing the number of samples by each class"
[image2]: ./figures/validation_distr.png "Class distribution in the validation set showing the number of samples by each class"
[image3]: ./figures/test_distr.png "Class distribution in the test set showing the number of samples by each class"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

Overview
---
In this project, we will use what we've learned about deep neural networks and convolutional neural networks to classify traffic signs. We will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, we will then try out our model on images of German traffic signs that we find on the web.

The following [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) was used as started point for our own implementation.

The main files of this project are:
* the Ipython notebook with the code
* the code exported as an html file
* this detailed documentation

The Project
---
The goals / steps of this project are the following:
* Load the German Traffic Sign data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results in this documentation

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details. We used Docker and all worked perfectly. 

### Dataset and Repository

1. Download the [data set](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which images were resized to 32x32. It contains a training, validation and test set.

2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

---
### Data Set Summary & Exploration

#### 1. Data Set Summary

We used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32,32,3)**
* The number of unique classes/labels in the data set is **43** which are the following:

<center>
  
Class Id|                               Sign Name
|:----:|:------------------------------------------------:| 
0      |                               Speed limit (20km/h)
1      |                              Speed limit (30km/h)
2      |                              Speed limit (50km/h)
3      |                              Speed limit (60km/h)
4      |                              Speed limit (70km/h)
5      |                              Speed limit (80km/h)
6      |                       End of speed limit (80km/h)
7      |                             Speed limit (100km/h)
8      |                             Speed limit (120km/h)
9      |                                        No passing
10     |      No passing for vehicles over 3.5 metric tons
11     |             Right-of-way at the next intersection
12     |                                     Priority road
13     |                                             Yield
14     |                                              Stop
15     |                                       No vehicles
16     |          Vehicles over 3.5 metric tons prohibited
17     |                                          No entry
18     |                                   General caution
19     |                       Dangerous curve to the left
20     |                      Dangerous curve to the right
21     |                                      Double curve
22     |                                        Bumpy road
23     |                                     Slippery road
24     |                         Road narrows on the right
25     |                                         Road work
26     |                                   Traffic signals
27     |                                       Pedestrians
28     |                                 Children crossing
29     |                                 Bicycles crossing
30     |                                Beware of ice/snow
31     |                             Wild animals crossing
32     |               End of all speed and passing limits
33     |                                  Turn right ahead
34     |                                   Turn left ahead
35     |                                        Ahead only
36     |                              Go straight or right
37     |                               Go straight or left
38     |                                        Keep right
39     |                                         Keep left
40     |                              Roundabout mandatory
41     |                                 End of no passing
42     | End of no passing by vehicles over 3.5 metric ...

</center>

We have tested if the training, validation and test sets contains samples of each class which is necessary because we must train, validate and test with samples from each class. In this case, samples are correctly distributed over all sets.

#### 2. Data Set Exploratory Visualization

We analyzed the class distribution in each split of the data set. At the same time we randomly plot a few examples of each split alongside its label. We will firstly plot several bar charts showing the class distribution on the training, validation and test sets.

![Class distribution in the training set showing the number of samples by each class][image1]
![Class distribution in the validation set showing the number of samples by each class][image2]
![Class distribution in the test set showing the number of samples by each class][image3]

Regarding the per-class occurrences in each set we noticed that we are facing a quite imbalanced data set. However validation and test sets are more balanced. 


