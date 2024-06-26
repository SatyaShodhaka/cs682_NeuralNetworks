from operator import index
import numpy as np
from scipy import stats

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    #created a dists matrix with all zeros which needs to be updated after
    #calculating the euclidean dist 
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        # calculating the L2 distance for each test image against all the
        # training examples

        # Test image dims = 500 * 3072 each row corresponds to an image and 
        # the columns are pixels
      
        # The nn classifier works by finding out the image(s) whose pixel
        # difference is low

        dist = np.sqrt(np.sum(np.square(np.abs(X[i]-self.X_train[j]))))
        dists[i,j] = dist

        pass
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      # dist = np.sqrt(np.sum(np.square(np.abs(X[i]-self.X_train)),axis = 1))
      # This approach above is computationally extensive and np.abs is redundant
      dist = np.sqrt(np.sum((X[i]-self.X_train)**2,axis=1))
      dists[i, :] = dist
      pass
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################


    # To Do
    # (A-B)**2 = A**2 + B**2 - 2* A.B
    # Add a new axis to X to allow broadcasting
    # 500x5000
    
    
    # square the pixels of each image and adding these squares along row 
    # axis = 1 such that it results in an array list with 500 elements where element is the 
    # sum of squares of all pixels of an Image

    XSquare = (X**2).sum(axis=1) #500x1 matrix

    # same logic to the train set but store it in a different variable and not 
    # mess with class object
    XTrainSquare = (self.X_train**2).sum(axis=1) #5000x1 matrix 
    
    DotOfXandX_train = X.dot(self.X_train.T)
    DotOfXandX_train = 2*DotOfXandX_train

    # Now to perform array broadcast sum on [500,] and [5000,]
    # add a new axis to XSquare so np can perform broadcast along that axis
    XSquare_reshape = XSquare[:, np.newaxis]
    XSquare = XSquare[:, np.newaxis]
    # print(XSquare.shape)
    
    dists = np.sqrt(XSquare_reshape + XTrainSquare - DotOfXandX_train)
    # print(XSquare.shape)
    pass
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      indexOfKNN = np.argsort(dists[i,:]) #1x5000 matrix
      # Get the K indexes
      indexOfKNN = indexOfKNN[:k]
      #contains indices of top k train images with low score
    
      # got the indexes of nearest K neighbours
      # Now get the J out of [i,j] which corresponds to the Train image

      pass
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      for x in indexOfKNN:
        closest_y.append(self.y_train[x])

      # get the most repeated label
      # result = stats.mode(closest_y) 
      # y_pred[i] = result[0] giving wrong output????
      y_pred[i] = max(closest_y, key = closest_y.count)

      pass
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################
  
    return y_pred

