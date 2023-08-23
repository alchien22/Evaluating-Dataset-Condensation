# Using LeNet-5 to Evaluate Distilled MNIST Images

## Method
The neural network architecture used is the LeNet-5 model with the following layers in order:
1. 2D Convolution
2. 2D Average Pooling
3. 2D Convolution
4. 2D Average Pooling
5. 2D Convolution
6. Flatten
- Flatten() is used to flatten the data before feeding it to a dense layer, which requires a 1D input
7. Fully-connected/Dense
- An L2 regularizer with value 0.1 is applied to reduce potential overfitting
- Through trial and error, it was determined that a larger lambda would have resulted in underfitting, which is why 0.1 was chosen
8. Fully-connected/Dense

The loss function used is SparseCategoricalCrossentropy or multiclass logistic loss.
The Adam optimizer is also used to speed up gradient descent.

The ReLU activation function is used for the hidden layers instead of the original tanh function as ReLU is typically more time efficient.
In order to obtain a more numerically accurate version of the loss function, a linear activation function is used for the output layer with the **SparseCategoricalCrossentropy** parameter **from_logits** set equal to **True**.
The probability distribution is then obtained by performing Softmax on the resulting logits.

### Distilled Datasets Used:  [[URL]](https://github.com/VICO-UoE/DatasetCondensation/tree/master)
- res_DC_MNIST_ConvNet_1ipc.pt
- res_DC_MNIST_ConvNet_10ipc.pt

### Citations
@article{RizwanLeNet-5
author={Rizwan, Muhammad},
title={LeNet-5-A Classic CNN Architecture},
year={2018},
url={https://www.datasciencecentral.com/lenet-5-a-classic-cnn-architecture/}
urldate={2023-8-22}
}

@misc{DistilledPatrick,
  title        = {Dataset Condensation},
  author       = {PatrickZH},
  year         = {2021},
  howpublished = {\url{https://github.com/VICO-UoE/DatasetCondensation}},
  note         = {Accessed: 2023-08-22},
}
