import numpy as np
import torch
import tensorflow as tf
from model import lenet_5
import logging
from time import time

#Now let's import non-distilled images from the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_count = 300
test_count = 600
train_images = train_images[:train_count]
train_labels = train_labels[:train_count]
test_images = test_images[:test_count]
test_labels = test_labels[:test_count]

height, width = 28, 28
channels = 1

#Set global seed so that results don't vary across runs
tf.random.set_seed(1234)

model = lenet_5.compile(height, width, channels)

#To prevent an excessive messages when running
logging.getLogger("tensorflow").setLevel(logging.ERROR)

start = time()

tf.keras.backend.clear_session()
lenet_5.train(model, train_images, train_labels, 300, None, None)

#Calculate training duration
duration = time() - start

lenet_5.summary(model)

#Model trained on normal MNIST data
#Tested on normal MNIST data

training_cerr, test_cerr = lenet_5.evaluate(model, train_images, train_labels, test_images, test_labels)

print(f"Training Accuracy     (regularized, non-distilled, 300 images): {1-training_cerr:0.7f}" )
print(f"Test Accuracy         (regularized, non-distilled, 600 images): {1-test_cerr:0.7f}" )
print(f"Time to Train         (regularized, non-distilled, 300 images): {duration:0.5f}" )