import numpy as np
import torch
import tensorflow as tf
from model import lenet_5
import logging
from time import time

input = torch.load('/Users/alexchien/Desktop/res_DC_MNIST_ConvNet_1ipc.pt')        #<------Use filepath to the 1ipc .pt file

#Check visualization of a single synthetic set without labels
# print(input['data'][0][0])

#Reshape tensor objects from batch_size, channels, height, width -> batch_size, height, width, channels using torch.permute() and numpy
#Create list of tensor objects without labels
#Framework conflict: Must convert PyTorch tensors into TensorFlow tensors by converting list into numpy array then converting to TF tensors
X=[]
for i in range(5):
    permuted = input['data'][i][0].permute(0, 2, 3, 1)
    numpy_X = permuted.numpy()
    tf_X = tf.convert_to_tensor(numpy_X, dtype = tf.float32)
    X.append(tf_X)

#Split image data into 60, 20, 20 for training, testing, and cross-validation sets
X_train = X[:3]
X_cv = X[3]
X_test = X[4]

#Extract input_shape of a single image: (height, width, channels)
#Batch size = 1ipc * 10 classes
batch_size, height, width, channels = X_train[0].shape
print('Batch_size: ', batch_size, ' Height: ', height, ' Width: ', width, ' Channels: ', channels)

#Create list of label tensor objects
#Convert PyTorch tensors to TensorFlow tensors
y=[]
for i in range(5):
    numpy_y = input['data'][i][1].numpy()
    tf_y = tf.convert_to_tensor(numpy_y, dtype = tf.float32)
    y.append(tf_y)

#Split labels into 60, 20, 20 for training, testing, and cross-validation sets
y_train = y[:3]
y_cv = y[3]
y_test = y[4]

#To prevent an excessive messages when running
logging.getLogger("tensorflow").setLevel(logging.ERROR)

#Set global seed so that results don't vary across runs
tf.random.set_seed(1234)

model = lenet_5.compile(height, width, channels)

#Combine input tensors since model expects 1 input tensor (simplifies process so we can treat input as single entity instead of multiple for each tensor)
#Initialize lists so that their values are reset every time the Jupyter notebook is run
X_train_combined=[]
y_train_combined=[]
X_temp=X_train[0]
y_temp=y_train[0]

#Essentially a running sum where you add the ith + 1 element in each iteration until reaching the end of the list
for i in range(len(X_train)):
    try:
        X_temp = np.concatenate((X_temp, X_train[i+1]), axis=0)
        y_temp = np.concatenate((y_temp, y_train[i+1]), axis=0)
    except:
        X_train_combined = X_temp
        y_train_combined = y_temp
        break

start = time()

tf.keras.backend.clear_session()
lenet_5.train(model, X_train_combined, y_train_combined, 30, X_cv, y_cv)

#Calculate training duration
duration = time() - start

#Model trained on 1ipc distilled MNIST data
#Tested on normal MNIST data

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
test_count = 600
test_images = test_images[:test_count]
test_labels = test_labels[:test_count]

training_cerr, test_cerr = lenet_5.evaluate(model, X_train_combined, y_train_combined, test_images, test_labels)

print(f"Training Accuracy    (regularized, distilled: 1ipc, 30 images): {1-training_cerr:0.7f}" )
print(f"Test Accuracy        (regularized, distilled: 1ipc, 10 images): {1-test_cerr:0.7f}" )
print(f"Time to Train        (regularized, distilled: 1ipc, 30 images): {duration:0.5f}" )