import numpy as np
from files.linear_svm import *

class LinearSVM:
  def __init__(self):
    self.W = None

  # returns (loss, grad)
  def loss(self, X_batch, y_batch, reg):
    loss = None
    grad = None
    #####################################
    # Your Code Here
    # Hint: reuse the functions you've already implemented in linear_svm

    loss, grad = svm_loss_vectorized(X_batch, y_batch, reg)
    
    #####################################
    return loss, grad


  # Returns an array of loss values per iteration
  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, 
    batch_size=200):

    dim, num_train = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    
    if self.W is None:
      self.W = np.random.randn(num_classes, dim) * 0.001

    loss_history = []
    for it in range(num_iters):
      X_batch = None
      y_batch = None

      indicies = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[:, indicies]
      y_batch = y[indicies]
      
      #####################################
      # Your Code Here

      # Hint: calculate loss e.g. loss, grad = ....
      # then add loss to history
      
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      #####################################

      if it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      #####################################
      # Your Code Here

      # Hint: Update self.W by calculated gradients. Remember to apply learning rate
      
      self.W -= learning_rate * grad


      #####################################

    return loss_history

  def predict(self, X):
    y_pred = np.zeros(X.shape[1])

    #####################################
    # Your Code Here

    predictions = W.dot(X)
    y_pred = np.argmax(predictions, axis=0)

    #####################################

    return y_pred