import numpy as np
import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import logging
from time import time
from functions import Functions

class lenet_5():
    def compile(height, width, channels):
        height = height
        width = width
        channels = channels
        model = Sequential(
            [
            #conv layer 1 (relu)
            Conv2D(input_shape = (height, width, channels), filters=6, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1'),
            #avg pooling
            AveragePooling2D(pool_size=2, strides=2, name='pooling1'),
            #conv layer 2 (relu)
            Conv2D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv2'),
            #avg pooling
            AveragePooling2D(pool_size=2, strides=2, name='pooling2'),
            #conv layer 3 (relu)
            Conv2D(filters=120, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv3'),
            #flatten
            Flatten(),
            #fully connected layer 1 (relu)
            Dense(84, activation='relu', name='dense1', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
            #fully connected layer 2 (linear)
            Dense(10, activation='linear', name='dense2')
            ], name='LeNet-5'
        )
        # filters are the same as output channel

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-2),
            metrics=['accuracy']
        )
        return model

    def train(model, train_images, train_labels, epochs, cv_images, cv_labels):
        if cv_images == None and cv_labels == None:
            model.fit(train_images, train_labels, epochs=epochs)
        else:
            model.fit(train_images, train_labels, epochs=epochs, validation_data=(cv_images, cv_labels))

    def evaluate(model, train_images, train_labels, test_images, test_labels):
        #make a model for plotting routines to call
        model_predict = lambda Xl: np.argmax(tf.nn.softmax(model.predict(Xl)).numpy(),axis=1)

        training_cerr = Functions.eval_cat_err(train_labels, model_predict(train_images))
        test_cerr = Functions.eval_cat_err(test_labels, model_predict(test_images))
        return training_cerr, test_cerr
    
    def summary(model):
        model.summary()

