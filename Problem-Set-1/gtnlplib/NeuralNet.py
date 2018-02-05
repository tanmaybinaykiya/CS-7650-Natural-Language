# -*- coding: utf-8 -*-
"""
Neural Networks
===============

Neural networks can be constructed using the ``torch.nn`` package.

Now that you had a glimpse of ``autograd``, ``nn`` depends on
``autograd`` to define models and differentiate them.
An ``nn.Module`` contains layers, and a method ``forward(input)``\ that
returns the ``output``.

For example, look at this network that classfies digit images:

.. figure:: /_static/img/mnist.png
   :alt: convnet

   convnet

It is a simple feed-forward network. It takes the input, feeds it
through several layers one after the other, and then finally gives the
output.

A typical training procedure for a neural network is as follows:

- Define the neural network that has some learnable parameters (or
  weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule:
  ``weight = weight - learning_rate * gradient``

Define the network
------------------

Let’s define this network:
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt

from gtnlplib import preproc, clf_base, constants


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(4875, 4, 100)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, self.num_flat_features(x))
        x = torch.tanh(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x
    #
    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features


# net = Net()
# print(net)

########################################################################
# You just have to define the ``forward`` function, and the ``backward``
# function (where gradients are computed) is automatically defined for you
# using ``autograd``.
# You can use any of the Tensor operations in the ``forward`` function.
#
# The learnable parameters of a model are returned by ``net.parameters()``

# params = list(net.parameters())
# print(len(params))
# print(params[0].size())  # conv1's .weight

########################################################################
# The input to the forward is an ``autograd.Variable``, and so is the output.
# Note: Expected input size to this net(LeNet) is 32x32. To use this net on
# MNIST dataset,please resize the images from the dataset to 32x32.

# input = Variable(torch.randn(1, 4875))
# out = net(input)
# print(out)

########################################################################
# Zero the gradient buffers of all parameters and backprops with random
# gradients:
# net.zero_grad()
# out.backward(torch.randn(1, 4))

########################################################################
# .. note::
#
#     ``torch.nn`` only supports mini-batches The entire ``torch.nn``
#     package only supports inputs that are a mini-batch of samples, and not
#     a single sample.
#
#     For example, ``nn.Conv2d`` will take in a 4D Tensor of
#     ``nSamples x nChannels x Height x Width``.
#
#     If you have a single sample, just use ``input.unsqueeze(0)`` to add
#     a fake batch dimension.
#
# Before proceeding further, let's recap all the classes you’ve seen so far.
#
# **Recap:**
#   -  ``torch.Tensor`` - A *multi-dimensional array*.
#   -  ``autograd.Variable`` - *Wraps a Tensor and records the history of
#      operations* applied to it. Has the same API as a ``Tensor``, with
#      some additions like ``backward()``. Also *holds the gradient*
#      w.r.t. the tensor.
#   -  ``nn.Module`` - Neural network module. *Convenient way of
#      encapsulating parameters*, with helpers for moving them to GPU,
#      exporting, loading, etc.
#   -  ``nn.Parameter`` - A kind of Variable, that is *automatically
#      registered as a parameter when assigned as an attribute to a*
#      ``Module``.
#   -  ``autograd.Function`` - Implements *forward and backward definitions
#      of an autograd operation*. Every ``Variable`` operation, creates at
#      least a single ``Function`` node, that connects to functions that
#      created a ``Variable`` and *encodes its history*.
#
# **At this point, we covered:**
#   -  Defining a neural network
#   -  Processing inputs and calling backward.
#
# **Still Left:**
#   -  Computing the loss
#   -  Updating the weights of the network
#
# Loss Function
# -------------
# A loss function takes the (output, target) pair of inputs, and computes a
# value that estimates how far away the output is from the target.
#
# There are several different
# `loss functions <http://pytorch.org/docs/nn.html#loss-functions>`_ under the
# nn package .
# A simple loss is: ``nn.MSELoss`` which computes the mean-squared error
# between the input and the target.
#
# For example:

# output = net(input)
# target = Variable(torch.arange(1, 5))  # a dummy target, for example
# criterion = nn.MSELoss()
#
# loss = criterion(output, target)
# print(loss)

########################################################################
# Now, if you follow ``loss`` in the backward direction, using it’s
# ``.grad_fn`` attribute, you will see a graph of computations that looks
# like this:
#
# ::
#
#     input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#           -> view -> linear -> relu -> linear -> relu -> linear
#           -> MSELoss
#           -> loss
#
# So, when we call ``loss.backward()``, the whole graph is differentiated
# w.r.t. the loss, and all Variables in the graph will have their
# ``.grad`` Variable accumulated with the gradient.
#
# For illustration, let us follow a few steps backward:
#
# print(loss.grad_fn)  # MSELoss
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

########################################################################
# Backprop
# --------
# To backpropagate the error all we have to do is to ``loss.backward()``.
# You need to clear the existing gradients though, else gradients will be
# accumulated to existing gradients
#
#
# Now we shall call ``loss.backward()``, and have a look at conv1's bias
# gradients before and after the backward.


# net.zero_grad()     # zeroes the gradient buffers of all parameters
#
# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)
#
# loss.backward()
#
# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)

########################################################################
# Now, we have seen how to use loss functions.
#
# **Read Later:**
#
#   The neural network package contains various modules and loss functions
#   that form the building blocks of deep neural networks. A full list with
#   documentation is `here <http://pytorch.org/docs/nn>`_
#
# **The only thing left to learn is:**
#
#   - updating the weights of the network
#
# Update the weights
# ------------------
# The simplest update rule used in practice is the Stochastic Gradient
# Descent (SGD):
#
#      ``weight = weight - learning_rate * gradient``
#
# We can implement this using simple python code:
#
# .. code:: python
#
#     learning_rate = 0.01
#     for f in net.parameters():
#         f.data.sub_(f.grad.data * learning_rate)
#
# However, as you use neural networks, you want to use various different
# update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc.
# To enable this, we built a small package: ``torch.optim`` that
# implements all these methods. Using it is very simple:


def run_optimizer(x, y, num_its=200, param_file='best.params'):
    this_net = Net()

    import torch.optim as optim

    # create your optimizer
    optimizer = optim.SGD(this_net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # in your training loop:
    for epoch in range(num_its):
        for index, inp in enumerate(x):
            target = y[index]

            optimizer.zero_grad()   # zero the gradient buffers
            output = this_net(inp)
            # print("O", output, "T", target)
            loss = criterion.forward(output, target)
            # print("Loss: ", loss)
            # print('conv1.bias.grad before backward. loss:', loss)
            # print(this_net.fc1.bias.grad)
            loss.backward()
            # print('conv1.bias.grad after backward')
            # print(this_net.fc1.bias.grad)
            optimizer.step()    # Does the update
            # if index > 5:
            #     break

    return this_net

###############################################################
# .. Note::
#
#       Observe how gradient buffers had to be manually set to zero using
#       ``optimizer.zero_grad()``. This is because gradients are accumulated
#       as explained in `Backprop`_ section.
#
###############################################################


def vectorize_y(y):
    unique_sorted_ys, indices = np.unique(y, return_inverse=True)
    y_vec = (np.arange(len(unique_sorted_ys)) == indices[:, np.newaxis]) + 0
    return y_vec, unique_sorted_ys


def vectorize_y_val(y):
    unique_sorted_ys, indices = np.unique(y, return_inverse=True)
    return indices


# def unvectorize_y(y_indices, unique_sorted_ys):
#     predicted_index = np.argmax(y_indices, axis=1)
#     return [unique_sorted_ys[i] for i in predicted_index]


def unvectorize_y(y_indices, unique_sorted_ys):
    return np.argmax(y_indices, axis=1)


def predict(x, t_net):
    output = np.zeros((x.shape[0], 4))
    for i, xi in enumerate(x):
        output[i] = t_net.forward(xi)
    return output


def runz(X_training_torch, Y_training_torch):

    # print("Initializing... ")
    # y_training, x_training = preproc.read_data('../lyrics-train.csv', preprocessor=preproc.bag_of_words)
    # y_dev, x_dev = preproc.read_data('../lyrics-dev.csv', preprocessor=preproc.bag_of_words)
    # # y_test, x_test = preproc.read_data('lyrics-test-hidden.csv', preprocessor=preproc.bag_of_words)
    # print("Import complete... ")
    #
    # counts_tr = preproc.aggregate_counts(x_training)
    # print("Pruning now... ")
    # x_training_pruned, vocab = preproc.prune_vocabulary(counts_tr, x_training, 10)
    # x_dev_pruned, _ = preproc.prune_vocabulary(counts_tr, x_dev, 10)
    # # x_test_pruned, _ = preproc.prune_vocabulary(counts_tr, x_test, 10)
    # print("Prune complete... ")
    #
    # print("Vectorizing... ")
    # X_training = preproc.make_numpy(x_training_pruned, vocab)
    # X_dev = preproc.make_numpy(x_dev_pruned, vocab)
    # # X_test = preproc.make_numpy(x_test_pruned, vocab)
    #
    # y_training, unique_sorted_ys = vectorize_y(y_training)
    # print("Vectorizing complete... ")
    #
    # print("NP - TORCH... ")
    # X_training_torch = Variable(torch.from_numpy(X_training.astype(np.float32)))
    # Y_training_torch = Variable(torch.from_numpy(y_training.astype(np.float32)))
    # X_dev_torch = Variable(torch.from_numpy(X_dev.astype(np.float32)))
    # print("NP - TORCH complete... ")

    print("Building model... ")
    this_net = run_optimizer(X_training_torch, Y_training_torch)
    print("Building model complete... ")

    print("Predicting... ")
    y_predicted = predict(X_training_torch, this_net)
    # y_target = vectorize_y_val(y_training)
    print("Prediction complete:... Accuracy: ", np.sum(y_predicted == Y_training_torch))

runz()

# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import GridSearchCV
#
# def scikit_nn():
#
#
#     print("Initializing... ")
#     y_training, x_training = preproc.read_data('../lyrics-train.csv', preprocessor=preproc.bag_of_words)
#     y_dev, x_dev = preproc.read_data('../lyrics-dev.csv', preprocessor=preproc.bag_of_words)
#     _, x_test = preproc.read_data('../lyrics-test-hidden.csv', preprocessor=preproc.bag_of_words)
#     print("Import complete... ")
#
#     counts_tr = preproc.aggregate_counts(x_training)
#     print("Pruning now... ")
#     x_training_pruned, vocab = preproc.prune_vocabulary(counts_tr, x_training, 10)
#     x_dev_pruned, _ = preproc.prune_vocabulary(counts_tr, x_dev, 10)
#     x_test_pruned, _ = preproc.prune_vocabulary(counts_tr, x_test, 10)
#     print("Prune complete... ")
#
#     print("Vectorizing... ")
#     X_training = preproc.make_numpy(x_training_pruned, vocab)
#     X_dev = preproc.make_numpy(x_dev_pruned, vocab)
#     X_test = preproc.make_numpy(x_test_pruned, vocab)
#
#     Y_training, unique_sorted_ys = vectorize_y(y_training)
#
#     nn_y_training = vectorize_y_val(y_training)
#     nn_y_dev = vectorize_y_val(y_dev)
#
#     print("Vectorizing complete... ", X_training.shape, nn_y_training.shape)
#
#     # parameters = {'solver': ('lbfgs', 'sgd', 'adam'), 'learning_rate_init': [0.01, 0.001, 0.0001],
#         'nesterovs_momentum':[True, False], 'activation': ['identity', 'logistic', 'tanh', 'relu']}
#     # gridsfit = GridSearchCV(MLPClassifier(), param_grid=parameters, verbose=10)
#     # gridsfit.fit(X_training, nn_y_training)
#     # gridsfit.score(X_dev, nn_y_dev)
#     # print("Acc: ", gridsfit.cv_results_, gridsfit.best_params_, gridsfit.best_score_)
#
#     mlp = MLPClassifier(solver='sgd', activation='logistic', learning_rate_init=0.001, max_iter=500, verbose=10)
#     mlp.fit(X_training, y_training)
#     score = mlp.score(X_dev, y_dev)
#     print("Y_dev score", score)
#
#     y_test_predicted = mlp.predict(X_test)
#     collect_bakeoff(y_test_predicted)
#
#
# def collect_bakeoff(labels):
#     predLabel = pd.DataFrame(labels)
#     predLabel.index += 1
#     predLabel.index.name = 'ID'
#     predLabel.columns = ['Era']
#
#     predLabel.to_csv("bakeoff_submission2.csv")

# scikit_nn()
