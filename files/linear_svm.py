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
  '''
  NOTE: we need to get this column_count b/c if you look at the derivation
  (or the code in svm_loss_naive) you'll notice that the derivation of W w.r.t
  y_i is applied for each class j that contributed to the error. 
  Therefore when we sum down with "axis=0" we're counting the number of classes
  that contributed to the loss.
  
  (Remember that 'binary' came from 'scores' which is in the shape of 10x300. 
  Each row is a class, each and column is an image therefore the intersect is 
  the image score. But you'll notice that we've modifed 'binary' to say which 
  image-class intersects contributed to the loss.)
  '''
  column_count = np.sum(binary, axis=0)

  '''
  Once we have column_count, we set the index of the 'correct_score' 'column_count'
  so that when we do the dot product below, 'dW = np.dot(binary, X.T)', we're
  effectively contributing X[:, i] the correct number of times. (Study this line of code
  'dW[y[i]] -= X[:, i]' in svm_loss_naive if it's still confusing. )
  '''
  binary[y, range(N)] = -column_count

  dW = np.dot(binary, X.T)

  dW /= N
  dW += reg * W

  loss /= N
  loss += 0.5 * reg * np.sum(W * W)
  

  #####################################

  return loss, dW;