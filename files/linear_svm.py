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


  
  dW /= num_images
  dW += reg * W
  loss /= num_images
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

  N = len(y)

  scores = W.dot(X)
  correct_scores = scores[y, range(N)]
  margin = scores - correct_scores + delta

  margin[y, range(N)] = 0

  margin = np.maximum(np.zeros_like(margin), margin)

  loss = np.sum(margin)

  binary = margin
  binary[binary > 0] = 1
  column_count = np.sum(binary, axis=0)

  binary[y, range(N)] = -column_count

  dW = np.dot(binary, X.T)

  dW /= N
  dW += reg * W

  loss /= N
  loss += 0.5 * reg * np.sum(W * W)
  

  #####################################

  return loss, dW;