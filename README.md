# <div align="center">**Dog and cat image recognition based on convolutional neural network**

## Table of contents

## 1.Introduction
 In this project, deep learning approach is used to build a convolutional neural network to achieve binary classification. 
 CNN can take an image as input and learn features of the image, and classify based on the learned characteristics.
 Then we train and evaluate the classifier on Kaggle’s dog vs. cat data set.

## 2.Tools & Installation
 The main tool we use in this project is the Keras library in python, which is an open source artificial neural network library written in Python. keras is built on top of tensorflow, so you need to install tensorflow before installing keras.
### 2.1
 

## 3.Data & Preparing
### Dataset source
 The dataset named Dogs vs. Cats used in this project is from Kaggle website. The dataset is available at here: [link](https://www.kaggle.com/competitions/dogs-vs-cats/data)
 The data is divided into a training set and a test set.Only the images in the train set have the label cat or dog in their names, so here we just use the images in train set.

### Data preparation
 We will take out some of the images for use in the model and divide them into the correct folders.
  **Firstly**, we create three folders with the names "train", "validation" and "test".
 
    my_dir = !pwd
    base_dir = my_dir[0] + '/dog_cat '
    os.mkdir(base_dir)
    train_dir = os.path.join(base_dir,"train")
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir,"validation")
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir,"test")
    os.mkdir(test_dir)
 _The created folders are as follows：_

 ![1](https://github.com/ACM40960/project-20211696/blob/main/images/2.jpeg)

  **Secondly**, under each folder we create two new folders names "dog" and "cat".

    train_cats_dir = os.path.join(train_dir, "cats")
    os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir, "dogs")
    os.mkdir(train_dogs_dir)
    validation_cats_dir = os.path.join(validation_dir, "cats")
    os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, "dogs")
    os.mkdir(validation_dogs_dir)
    test_cats_dir = os.path.join(test_dir, "cats")
    os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir, "dogs")
    os.mkdir(test_dogs_dir)

  _The created folders are as follows：_

 ![2](https://github.com/ACM40960/project-20211696/blob/main/images/1.png)

  **At last**，we will put the images we need into these files.

 ![3](https://github.com/ACM40960/project-20211696/blob/main/images/3.png)