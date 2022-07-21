import os
import tensorflow as tf
import math
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPool1D, ReLU, Dropout, Softmax, concatenate, Flatten, Reshape, \
    GaussianNoise,Activation
from keras.layers.convolutional import Conv2D
from keras.layers import CuDNNLSTM,Lambda,Multiply,Add,Subtract,MaxPool2D,CuDNNGRU,LeakyReLU,BatchNormalization
import tensorflow as tf

def LSTM(weights=None,
           input_shape=[128,2],
           classes=6,
           **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5 # dropout rate (%)
    input = Input(input_shape, name='input1')
    x4= CuDNNLSTM(units=128, return_sequences=True)(input)
    x4 = CuDNNLSTM(units=128)(x4)
    x4 = Dropout(dr)(x4)
    x = Dense(classes, activation='softmax', name='softmax')(x4)

    model = Model(inputs = input, outputs=x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model