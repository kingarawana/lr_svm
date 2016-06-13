import numpy as np

def svm_loss_naive(W, X, y, reg, delta=1):
  loss = 0.0
  dW = np.zeros(W.shape)

  '''
  W = 10x3073
  X = 3073x300
  y = 300
  '''

  #####################################
  # Your Code Here


  num_classes = W.shape[0]
  num_images = len(y)

  for i in range(num_images):
    scores = W.dot(X[:, i]) # each row is a class score
    correct_score = scores[y[i]]
    for j in range(num_classes):
      if(j != y[i]):
        margin = scores[j] - correct_score + delta
        if(margin > 0):
          loss += margin

          dW[j] += X[:, i]
          dW[y[i]] -= X[:, i]


  
  dW += reg * W

  loss /= num_images
  dW /= num_images

  loss += 0.5 * reg * np.sum(W * W)

  

  #####################################

  return loss, dW;

def svm_loss_vectorized(W, X, y, reg, delta=1):
  loss = 0.0
  dW = np.zeros(W.shape)

  '''
  W = 10x3073
  X = 3073x300
  y = 300
  '''

  #####################################
  # Your Code Here



  #####################################

  return loss, dW;