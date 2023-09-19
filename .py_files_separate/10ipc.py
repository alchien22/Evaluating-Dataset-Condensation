import numpy as np
import torch
import tensorflow as tf
from model import lenet_5
import logging
from time import time

input_10 = torch.load('<--filepath-->')         #<------Use filepath to the 10ipc .pt file

# print(input_10['data'][0])

#Create list of tensor objects without labels
X_10=[]
for i in range(5):
    permuted = input_10['data'][i][0].permute(0, 2, 3, 1)
    numpy_X = permuted.numpy()
    tf_X = tf.convert_to_tensor(numpy_X, dtype = tf.float32)
    X_10.append(tf_X)

#Split image data into 60, 20, 20 for training, testing, and cross-validation sets
X10_train = X_10[:3]
X10_cv = X_10[3]
X10_test = X_10[4]

batch_size_10, height_10, width_10, channels_10 = X10_train[0].shape
print('Batch_size: ', batch_size_10, ' Height: ', height_10, ' Width: ', width_10, ' Channels: ', channels_10)

#Create list of label tensor objects
y_10=[]
for i in range(5):
    numpy_y = input_10['data'][i][1].numpy()
    tf_y = tf.convert_to_tensor(numpy_y, dtype = tf.float32)
    y_10.append(tf_y)

#Split labels into 60, 20, 20 for training, testing, and cross-validation sets
y10_train = y_10[:3]
y10_cv = y_10[3]
y10_test = y_10[4]

#To prevent an excessive messages when running
logging.getLogger("tensorflow").setLevel(logging.ERROR)

#Set global seed so that results don't vary across runs
tf.random.set_seed(1234)

model = lenet_5.compile(height_10, width_10, channels_10)

#Initialize lists so that their values are reset every time the Jupyter notebook is run
X10_train_combined=[]
y10_train_combined=[]
X_temp=X10_train[0]
y_temp=y10_train[0]

#Essentially a running sum where you add the ith + 1 element in each iteration until reaching the end of the list
for i in range(len(X10_train)):
    try:
        X_temp = np.concatenate((X_temp, X10_train[i+1]), axis=0)
        y_temp = np.concatenate((y_temp, y10_train[i+1]), axis=0)
    except:
        X10_train_combined = X_temp
        y10_train_combined = y_temp
        break


start = time()

tf.keras.backend.clear_session()
lenet_5.train(model, X10_train_combined, y10_train_combined, 300, X10_cv, y10_cv)

# calculate and report duration of concatenation
duration = time() - start

#Model trained on 10ipc distilled MNIST data
#Tested on normal MNIST data

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
test_count = 600
test_images = test_images[:test_count]
test_labels = test_labels[:test_count]

training_cerr, test_cerr = lenet_5.evaluate(model, X10_train_combined, y10_train_combined, test_images, test_labels)

print(f"Training Accuracy  (regularized, distilled: 10ipc, 300 images): {1-training_cerr:0.7f}" )
print(f"Test Accuracy      (regularized, distilled: 10ipc, 600 images): {1-test_cerr:0.7f}" )
print(f"Time to Train      (regularized, distilled: 10ipc, 300 images): {duration:0.5f}" )