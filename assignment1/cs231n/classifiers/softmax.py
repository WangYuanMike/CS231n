import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in xrange(num_train):
    # Compute loss for training sample i and update total loss
    score = np.dot(X[i],W)
    score -= np.max(score)  # Numeric stability
    exp_score = np.exp(score)
    softmax = exp_score / np.sum(exp_score)
    loss -= np.log(softmax[y[i]])

    # Compute gradient for training sample i and update total gradient
    dW[:,y[i]] += X[i,:] * (softmax[y[i]] - 1)    # correct class

    for j in xrange(num_classes):
      if j == y[i]:
        continue
      else:
        dW[:,j] += X[i,:] * softmax[j]    # other classes

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train        # Do it for the gradient matrix as well

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  loss = 0.0

  # Compute loss for training sample i and update total loss
  score = np.dot(X,W)
  score -= np.ones(score.shape) * np.max(score, axis=1)[:,np.newaxis]  # Numeric stability
  exp_score = np.exp(score)
  softmax = exp_score / np.sum(exp_score, axis=1)[:,np.newaxis]
  correct_softmax = softmax[np.arange(0, softmax.shape[0]), y]
  correct_log_softmax = np.log(correct_softmax)
  loss -= np.sum(correct_log_softmax)

  # Compute gradient for training sample i and update total gradient
  softmax[np.arange(0, softmax.shape[0]), y] -= 1
  dW = np.dot(X.T, softmax) / num_train

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

