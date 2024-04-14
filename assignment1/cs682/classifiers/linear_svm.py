import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in range(num_train):
    scores = X[i].dot(W)
    loss_classes = 0
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      # calculate the classes or weights which contributed to the loss func for grad
      # scores is WxX_i
      if margin > 0:
        dW[:,j] += X[i]
        loss += margin
        dW[:,y[i]] += -X[i]
      
    
    # gradient computation
         
        
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /=num_train


  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  # Adding dW regulatization 



  """
  To calculate the gradient dW matrix = loss(after modifying) each element in
  W and computing the loss. Then dW-loss gives the gradient for each weight w

  """

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  # calculate the scores
  scores = X.dot(W)

  
  """
  Algo:

  Scores matrix has the Wx values for each class row wise, each row stores
  10 scores for an image
  As we are also considering j=y_i while calculating the margin matrix
  > Iterate thought the label for each image in Y row wise and make the element
  zero
  > iterate through the scores matrix columwise and make the y[i]the row zero
  """
  margin = np.maximum(0,scores-scores[np.arange(len(y)),y][:,np.newaxis]+1)
  # broadcasting for sum
  # margin has sj-sy+1 using the broad cast pick the actual class score from
  # also as we are considering the actuall class and we need to set it to zero
  # as the margin for it set as 1 and we need to null it
  margin [np.arange(len(y)),y] = 0
  num_train = X.shape[0]
  loss = np.sum(margin)
  loss /= num_train
  loss += reg * np.sum(W * W)


  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  """
  From the score matrix find out the classes that contributed to the loss
  -> iterate through the matrix if the margin is > 0 then make it one
  -> this matrix.xi is the gradient for that row of class
  -> for the actual class -count(margins).xi
  """
  
  margin = np.where(margin>0,1,0) #if the margin is greater than 0 replace it with 1 
  sum_of_margin_classes = np.sum(margin, axis=1) #compute total 1s in a row and replace y_i with -count
  margin[np.arange(len(y)), y] = -sum_of_margin_classes 
  dW = (X.T).dot(margin)
  dW /= num_train
  dW += 2 * reg * W
 
 
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
