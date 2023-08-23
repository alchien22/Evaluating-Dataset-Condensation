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

input_10 = torch.load('/Users/alexchien/Downloads/res_DC_MNIST_ConvNet_10ipc.pt')   #<------Use filepath to the 10ipc .pt file

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

model_10 = Sequential(
    [
    #conv layer 1 (relu)
    Conv2D(input_shape = (height_10, width_10, channels_10), filters=6, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1'),
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

model_10.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-2),
    metrics=['accuracy']
)

#With 10ipc (images per class), we have batch sizes of 100 so the train set size is 300

start = time()

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

# calculate and report duration of concatenation
duration = time() - start
print(f'Took {duration:.5f} seconds')

model_10.fit(X10_train_combined, y10_train_combined, epochs=300, validation_data=(X10_cv, y10_cv))

# Calculate the categorization error
#y: target value
#yhat: predicted value
#cerr: % incorrect

def eval_cat_err(y, yhat):
    m = len(y)
    incorrect = 0
    for i in range(m):
       if yhat[i] != y[i]:
            incorrect+=1
    cerr = incorrect / m
    
    return(cerr)

#make a model for plotting routines to call
model_predict_10 = lambda Xl: np.argmax(tf.nn.softmax(model_10.predict(Xl)).numpy(),axis=1)

training_cerr_10 = eval_cat_err(y10_train_combined, model_predict_10(X10_train_combined))
cv_cerr_10 = eval_cat_err(y10_cv, model_predict_10(X10_cv))
test_cerr_10 = eval_cat_err(y10_test, model_predict_10(X10_test))
print(f"categorization error, training, regularized, 10ipc: {training_cerr_10:0.5f}" )
print(f"categorization error, cv,       regularized, 10ipc: {cv_cerr_10:0.5f}" )
print(f"categorization error, test,     regularized, 10ipc: {test_cerr_10:0.5f}" )