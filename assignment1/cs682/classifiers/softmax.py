import numpy as np
from random import shuffle

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

  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]


  for i in range(num_train):
    scores = X[i].dot(W)
    scores = scores - np.max(scores)
    scores = np.exp(scores)
    correct_class_score = scores[y[i]]
    sum_of_scores = np.sum(scores)
    probs = scores/sum_of_scores

  
    L_i = -np.log(correct_class_score/sum_of_scores)
    loss += L_i

    # Gradient calculation
    # probs have the score probabilities and the size is [10,1]
    # by the chain rule of differentiation we can compute the gradient
    # The gradient is given by Pj - 1(if j=y[i]) other wise Pj
    # from the list of probs do -1 on the correct class

    probs[y[i]] -= 1
    X_reshape = X[i][:,np.newaxis]
    probs_reshape = probs[np.newaxis, :]

    # dW 3073x10
    # probs = 10,
    # X_reshape = 3073
    dW += X_reshape.dot(probs_reshape)
    

  


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2* reg * W
  
  pass
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
  scores = X.dot(W)
  

  """
  1. Do exp on all scores
  2. For each score calculate score/sum of scores
  3. Take sum and divide on the whole matrix
  4. Sum of this matrix is the total loss
  """

  scores = scores - np.max(scores, axis=1, keepdims=True) #keepdims=True to preserve the rows
  scores = np.exp (scores)
  # calculate the sum of scores image wise
  sum_of_scores = np.sum(scores, axis = 1, keepdims=True)
  # go through each row of scores and return the currect class score
  # divide the correct class scores with sum of scores imagewise(by row)
  probs = scores/sum_of_scores
  L = -np.log(probs[np.arange(len(y)),y])
  # print(L.shape)
  # print(L[:5,])
  
  loss = np.sum(L)/X.shape[0] 

  loss += reg*np.sum(W*W)
  
  # gradient
  probs[np.arange(len(y)),y] -= 1
  # dW 3073x10
  # probs = 10,
  # X_reshape = 3073
  dW = X.T.dot(probs)
  dW  /= X.shape[0]
  dW += 2 * reg * W

  
  
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

