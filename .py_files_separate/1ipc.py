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

input = torch.load('/Users/alexchien/Desktop/res_DC_MNIST_ConvNet_1ipc.pt')          #<------Use filepath to the 10ipc .pt file

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

logging.getLogger("tensorflow").setLevel(logging.ERROR)

#Set global seed so that results don't vary across runs
tf.random.set_seed(1234)

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

start = time()

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

#Display duration
duration = time() - start
print(f'Took {duration:.5f} seconds')

model.fit(X_train_combined, y_train_combined, epochs=300, validation_data=(X_cv, y_cv))

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
model_predict = lambda Xl: np.argmax(tf.nn.softmax(model.predict(Xl)).numpy(),axis=1)

training_cerr = eval_cat_err(y_train_combined, model_predict(X_train_combined))
cv_cerr = eval_cat_err(y_cv, model_predict(X_cv))
test_cerr = eval_cat_err(y_test, model_predict(X_test))
print(f"categorization error, training, regularized, 1ipc: {training_cerr:0.7f}" )
print(f"categorization error, cv,       regularized, 1ipc: {cv_cerr:0.7f}" )
print(f"categorization error, test,     regularized, 1ipc: {test_cerr:0.7f}" )