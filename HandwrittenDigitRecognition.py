# a simple convnet on the MNIST dataset.

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

      # STEP 1: Load image data from MNIST.
      
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# we have 60,000 samples in our training set, 
# and the images are 28 pixels x 28 pixels each.
# the data, shuffled and split between train and test sets

# x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
# y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples).
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(mnist.head())

# print (x_train.shape)   #(60000, 28, 28)
# print (y_train.shape)   #(60000,)

      # STEP 2: Preprocess input data for Keras.

# Our MNIST images only have a depth of 1, 
# transform our dataset from having shape (n, width, height) 
# to (n, depth, width, height) or (n, width, height, depth)


# x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
# x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
# input_shape = (1, img_rows, img_cols)

# OR

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# print (x_train.shape)   #(60000, 1, 28, 28) or (60000, 28, 28, 1)

# The final preprocessing step for the input data is to convert our data type to 
# float32 and normalize our data values to the range [0, 1].
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')  #60,000
print(x_test.shape[0], 'test samples')    #10,000   

      # STEP 3: Preprocess class labels for Keras.

# print (y_train.shape)   #(60000,) 

# convert class vectors to binary class matrices...1D array to 10D matrix
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# print (y_train.shape)   #(60000,10) this is a matrix

      # STEP 4: Define model architecture.

model = Sequential()
# number of convolution filters to use = 32
# the number of rows and columns in each convolution kernel = 3,3
# Rectified Linear Unit: computes the function f(x)=max(0,x). In other words, 
#                        the activation is simply thresholded at zero
# The step size is (1,1) by default
# input shape parameter should be the shape of 1 sample, (1, 28, 28)
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

# print (model.output_shape)

# add more layers
model.add(Conv2D(64, (3, 3), activation='relu'))      
#  to reduce the number of parameters in our model by sliding a 2x2 pooling filter across the previous layer and 
# taking the max of the 4 values in the 2x2 filter.
model.add(MaxPooling2D(pool_size=(2, 2)))
# regularizing our model in order to prevent overfitting
model.add(Dropout(0.25))        #??????

#  we've added two Convolution layers

# fully connected layer (1-D)
model.add(Flatten())
# For FC Dense layers, the first parameter is the output size of the layer. 
# Keras automatically handles the connections between layers.
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))
# the final layer has an output size of 10, corresponding to the 10 classes of digits.
model.add(Dense(num_classes, activation='softmax'))

      # STEP 5:  Compile model.

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

      # STEP 6: Fit model on training data.

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

      # STEP 7: Evaluate model on test data.

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
