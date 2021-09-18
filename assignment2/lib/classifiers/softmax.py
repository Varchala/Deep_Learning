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
  # here, it is easy to run into numeric instability. So use stable version   #
  # of softmax. Don't forget the regularization!                              #
  #############################################################################
  num_train = X.shape[0]

  for i in xrange(num_train):
    scores = X[i].dot(W)
    # convert scores to probabilities of the classes by using stable version of softmax function
    C=max(scores)
    esy=np.exp(scores-C)
    Ees=np.sum(np.exp(scores-C))
    prob=esy/Ees
    loss =loss-np.log(prob[y[i]])

    #For computing gradient based on professor's notes
    prob[y[i]]=prob[y[i]]-1
    for j in xrange(W.shape[1]):
      dW[:,j] += X[i,:] * prob[j]

  #overall loss with regularization term as explained in slides
  loss=loss/num_train
  loss += 0.5*reg * np.sum(W * W)
  #overall grad
  dW = dW/num_train + reg*W
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
  scores = X.dot(W)
  C=np.max(scores,axis=1,keepdims=True)
  esy=np.exp(scores-C)
  Ees=np.sum(esy,axis=1,keepdims=True)
  prob=esy/Ees
  loss =np.sum(-np.log(prob[range(num_train),y]))
  loss = loss/num_train
  loss += 0.5 * reg * np.sum(W * W)

  #For computing gradient based on professor's notes
  prob[range(num_train),y]=prob[range(num_train),y]-1
  dW =X.T.dot(prob)
  dW = dW/num_train
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

