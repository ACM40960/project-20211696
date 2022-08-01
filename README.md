# <div align="center">**Dog and cat image recognition based on convolutional neural network**

![8](https://github.com/ACM40960/project-20211696/blob/main/images/8.jpeg)

## Table of contents

### [1.Introduction](https://github.com/ACM40960/project-20211696#1introduction)
### [2.Tools & Installation](https://github.com/ACM40960/project-20211696#2tools--installation)
### [3.Data & Preparing](https://github.com/ACM40960/project-20211696#3data--preparing)
### [4.Model construction & fit](https://github.com/ACM40960/project-20211696#4model-construction--fit)
### [5.Model evaluation](https://github.com/ACM40960/project-20211696#5model-evaluation)
### [Additional Statements](https://github.com/ACM40960/project-20211696#additional-statements)

## 1.Introduction
 In this project, deep learning approach is used to build a convolutional neural network to achieve binary classification. 
 CNN can take an image as input and learn features of the image, and classify based on the learned characteristics.
 Then we train and evaluate the classifier on Kaggle’s dog vs. cat data set.

## 2.Tools & Installation
 The main tool we use in this project is the Keras library in python, which is an open source artificial neural network library written in Python. keras is built on top of tensorflow, so you need to install tensorflow before installing keras.

### Editor
 The python editor I use is jupyter, it needs to be downloaded from Anaconda or Miniconda, you can go to the download page via the link.
 
 [Anaconda](https://www.anaconda.com/)
 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### TensorFlow
 First you need to check your python version and your computer system version,
 then go to the official tensorflow website[link](https://www.tensorflow.org/install/source) and choose the right tensorflow version to download.
 
_Check python version:_
 
![6](https://github.com/ACM40960/project-20211696/blob/main/images/6.png)

_Check laptop system_

![7](https://github.com/ACM40960/project-20211696/blob/main/images/7.png)

I chose the latest version of tensorflow, tensorflow-2.9.0.

To install it we need to type in computer base（not jupyter note!）. 

    python -m pip install tensorflow==2.9.0

Then we can type code below to check if it is successfully installed.

    import tensorflow as tf
    tf.__version__

### Other libraries

 We also need some other packages, we can import them directly via jupyter notebook.
    
    import cv2
    import os, shutil
    import tensorflow as tf
    from keras import layers 
    from keras import models
    from tensorflow.keras import optimizers
    from keras.preprocessing.image import ImageDataGenerator
    import PIL
    from PIL import Image
    import matplotlib.pyplot as plt



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
 
### Data pre-processing

 All images here are not of the same size and shape, 
so to prepare the model we need to resize the images to the same size square. 
Here images are reshaped to 180x180 pixels and rescaling by 255.

 Moreover, we'll use data augmentation on training data, we randomly flip half of the images horizontally, followed by panning the images horizontally and vertically.
 We also randomly scale the images, randomly stagger transform, and randomly rotate them. 

    test_datagen = ImageDataGenerator(rescale=1./255)  # for validation and test data 
    train_datagen =tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest",
    horizontal_flip=True)    # for train data

    train_generator = train_datagen.flow_from_directory(
    train_dir,  
    target_size=(180,180),  
    batch_size=50,
    class_mode="binary" )
    validation_generator = test_datagen.flow_from_directory(
    validation_dir, 
    target_size=(180,180),  
    batch_size=50,
    class_mode="binary"  )

## 4.Model construction & fit

### Model building

 In this part we specify a CNN model with four convolutional layers,
 interleaved with four pooling layers, and then followed three fully connected layers.
 The first convolution layer is set with 32 filters and a 3 × 3 kernel with default strides.
 The second convolution layer is set with 64 filters, with 3 × 3 kernels. 
 The following two convolution layers are set with 128 filters, with 3 × 3 kernels. All max-pooling layers have a pool size of 2 × 2,
 thus halving width and height at every pass. In the fully connected layers, we add dropout = 0.2.

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3),activation = "relu" , input_shape = (180,180,3)) ,
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64,(3,3),activation = "relu") ,  
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128,(3,3),activation = "relu") ,  
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128,(3,3),activation = "relu"),  
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dropout(0.2),
    
        tf.keras.layers.Dense(1000,activation="relu"),      #Adding the Hidden layer
    
    
        tf.keras.layers.Dense(100,activation="relu"),
        tf.keras.layers.Dense(1,activation = "sigmoid")   #Adding the Output Layer
    ])

 Then we further set the neural network parameters.Then we further set the neural network model parameters. We choose the optimizer RMSProp, 
 set the learning rate to be 0.0001, and choose binary_crossentropy as the loss function since it is a binary classification problem.
 
    from tensorflow.keras import optimizers
    model.compile(loss="binary_crossentropy",
             optimizer=optimizers.RMSprop(learning_rate=1e-4),
             metrics=["acc"])

### Model fitting

We used the processed training and validation data for model fitting, setting 100 epochs.

    history = model.fit(
    train_generator,
    epochs=100,  
    validation_data=validation_generator,  )

The model training process is visualized as follows：

    epochs = range(1, len(acc)+1)
    # acc
    plt.figure(figsize=(10, 6), dpi=80)
    plt.plot(epochs, acc, color='orange', linestyle=':', marker='.', markersize=7, label="Training acc")
    plt.plot(epochs, val_acc, color='blue', linestyle=':', marker='.', markersize=7, label="Validation acc")
    plt.title("Training and Validation acc")
    plt.legend()
    plt.grid(alpha=0.8)
    # loss
    plt.figure(figsize=(10, 6),dpi=80)
    plt.plot(epochs, loss, label="Training loss",color='orange', linestyle=':', marker='.', markersize=7)
    plt.plot(epochs, val_loss,label="Validation loss",color='blue', linestyle=':', marker='.', markersize=7)
    plt.title("Training and Validation loss")
    plt.legend()
    plt.grid(alpha=0.8)

**Accuracy**

![4](https://github.com/ACM40960/project-20211696/blob/main/images/4.jpeg)

**Loss**

![5](https://github.com/ACM40960/project-20211696/blob/main/images/5.jpeg)

Now we can see our classifier has an accuracy of approximately 82.99% and a loss of about 0.3841.
This is an acceptable result and we will further see how it performs on the test set.

## 5.Model evaluation
 After model fitting we will test our model on test data.
 **Firstly**, test images are reshaped to 180x180 pixels and rescaling by 255.
 
    test_generator = test_datagen.flow_from_directory(
    test_dir,  
    target_size=(180,180),  
    batch_size=20,
    class_mode="binary")

 **Next**, we evaluate the model.
  
    model.evaluate(test_generator)
 
 **Our test result are as follows：**

 <div align="center">

| Accuracy  | Loss|
| ---------- | -----------|
| 77.75%  | 0.4548   |

</div>

The results on the brand completely new test dataset are 77.75% accuracy and a loss of 0.4548, which means this model will help us to identify cats and dogs in a reliable way.

## 6.Additional Statements
 More details can be viewed in [project](https://github.com/ACM40960/project-20211696/blob/main/Final%20Report.pdf), please refer to the code details in project.ipynb, the total time of the code run is about one hour.
 
 Thanks for your reading!