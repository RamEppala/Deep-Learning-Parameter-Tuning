# Deep-Learning-Parameter-Tuning
project Name: Deep Learning and Parameter Tuning with MXnet, H2o Package in R

this network performs terribly on this data. In fact, it gives no better result than the train accuracy. On this data set, xgboost tuning gave 87% accuracy.

Description:

Build deep learning models in R using MXNet and H2O package. Also,tuned parameters of a deep learning model for better model performance.

Table of Contents:

What is Deep Learning ? How is it different from a Neural Network?

Deep Learning is the new name for multilayered neural networks. You can say, deep learning is an enhanced and powerful form of a neural network. The difference between the two is subtle.

The difference lies in the fact that, deep learning models are build on several hidden layers (say, more than 2) as compared to a neural network (built on up to 2 layers).

How does Deep Learning work ?

It works like this:

The dendrites receive the input signal (message).
These dendrites apply a weight to the input signal. Think of weight as "importance factor" i.e. higher the weight, higher the importance of signal.
The soma (cell body) acts on the input signal and does the necessary computation (decision making).
Then, the signal passes through the axon via a threshold function. This function decides whether the signal needs to be passed further.
If the input signal exceeds the threshold, the signal gets fired though the axon to terminals to other neuron.
This is a simplistic explanation of human neurons. The idea is to make you understand the analogy between human and artificial neurons

Why is bias added to the network ?

Bias (wo) is similar to the intercept term in linear regression. It helps improve the accuracy of prediction by shifting the decision boundary along Y axis.

What are activation functions and their types ?

activation functions govern the type of decision boundary to produce given a non-linear combination of input variables. Also, due to their mathematical properties, activation functions play a significant role in optimizing prediction accuracy.

Multi Layered Neural Networks

A multilayered neural network comprises a chain of interconnected neurons which creates the neural architecture. Along with input and output layers, it consists of multiple hidden layers also.

Multilayered neural networks are preferred when the given data set has a large number of features. That's why this model is being widely used to work on images, text data, etc. There are several types of neural networks; two of which are most commonly used:

Feedforward Neural Network: In this network, the information flows in one direction, i.e., from the input node to the output node.
Recurrent (or Feedback) Neural Network: In this network, the information flows from the output neuron back to the previous layer as well. It uses the backpropagation algorithm.

What is Backpropagation Algorithm ? How does it work ?

The goal of the backpropagation algorithm is to optimise the weights associated with neurons so that the network can learn to predict the output more accurately. Once the predicted value is computed, it propagates back layer by layer and re-calculates weights associated with each neuron. In simple words, it tries to bring the predicted value as close to the actual value.

Gradient Descent

gradient descent works like this:

First, it calculates the partial derivative of the weight.
If the derivative is positive, it decreases the weight value.
If the derivative is negative, it increases the weight value.
The motive is to reach to the lowest point (zero) in the convex curve where the derivative is minimum.
It progresses iteratively using a step size (η), which is defined by the user. But make sure that the step size isn't too large or too small. Too small a step size will take longer to converge, too large a step size will never reach an optimum.

Practical Deep Learning with H2O & MXnet

H2O package provides h2o.deeplearning function for model building. It is built on Java. Primarily, this function is useful to build multilayer feedforward neural networks. It is enabled with several features such as the following:

Multi-threaded distributed parallel computation
Adaptive learning rate (or step size) for faster convergence
Regularization options such as L1 and L2 which help prevent overfitting
Automatic missing value imputation
Hyperparameter optimization using grid/random search

parameters involved in model building with h2o.

hidden - It specifies the number of hidden layers and number of neurons in each layer in the architechture.
epochs - It specifies the number of iterations to be done on the data set.
rate - It specifies the learning rate.
activation - It specifies the type of activation function to use. In h2o, the major activation functions are Tanh, Rectifier, and Maxout.

MXNetR Package

The mxnet package provides an incredible interface to build feedforward NN, recurrent NN and convolutional neural networks (CNNs). CNNs are being widely used in detecting objects from images. CNNs are being widely used in detecting objects from images.
