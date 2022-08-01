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
