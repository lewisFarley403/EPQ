import tensorflow as tf
from tensorflow import keras
from keras import layers
import cv2  # for testing
import numpy as np
import matplotlib.pyplot as plt
FEATURE_SPACE = 512
IMAGE_SIZE = (120, 120, 3)


def encoder():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(120, activation='relu',
                            kernel_size=(2, 2), input_shape=IMAGE_SIZE))
    model.add(layers.Conv2D(240, activation='relu',
                            kernel_size=(3, 3)))
    model.add(layers.Dense(FEATURE_SPACE, activation='relu'))
    return model


def decoder():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2DTranspose(
        FEATURE_SPACE, kernel_size=(2, 2), input_shape=(1, 1, 512)))
    # model.add(keras.layers.Conv2DTranspose(
    #     480, kernel_size=(2, 2)))
    model.add(keras.layers.Conv2DTranspose(
        3, kernel_size=(3, 3), activation='sigmoid'))  # used the sigmoid activation function because pixels need to be between 0 and 1, relu sets all negative values to 0 else it grows linear with the output
    return model


def autoencoder():
    model = encoder()
    model.add(decoder())
    return model


e = encoder()
img = cv2.imread(r'C:\Users\lewis\OneDrive\Desktop\code\maskremove\roi\0.jpg')
print(img.shape)
v = e.predict(np.asarray([img]))
d = decoder()
print(d.predict(v).shape)
print(v.shape)
img = d.predict(v)

print(img[0].shape)
cv2.imshow('test', img[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
here is the encoder decoder network:
I started by defining some constants i needed for the program to work, like feature space (aka latent space) and the size of the image.

1/3/22
- built encoder, like image classifier 
- decoder literal opposite
- built first image
- some kind of error where the output image is 121x121 rather than 120x120, could be padding caused by the non exact fit of the conv2d layers
- the memory utilisation is high, 106,936,320 (assume this is in bytes but not concrete) with batch size of 1
2/3/22
- fixed the 121x121 error, due to a typo and an extra convolution layer
- unable to find faces with masks on, it breaks the face finder
4/3/22
- trying to find a way to identify faces with masks. Found website on a haar cascade, the most common way: https://medium.com/analytics-vidhya/haar-cascades-explained-38210e57970d 
  the article also inclues a reference to the origional paper https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf 
'''
