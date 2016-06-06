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


  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  loss /= num_images
  dW /= num_images



  

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

  num_classes = W.shape[0]
  num_images = X.shape[1]

  scores = W.dot(X) # 10x300
  correct_scores = scores[y, range(num_images)] # the correct scores for each of the 300 images
  margins = scores - correct_scores + delta

  margins[y, range(num_images)] = 0 # we have to zero out correct values b/c of + delta
  margins = np.maximum(np.zeros((num_classes, num_images)), margins)

  loss = np.sum(margins)
  loss += 0.5 * reg * np.sum(W * W)

  binary = margins
  binary[binary > 0] = 1
  column_count = np.sum(binary, axis=0)
  binary[y, range(num_images)] = -column_count
  dW = np.dot(binary, X.T)

  loss /= num_images
  dW /= num_images

  dW += reg * W




  

  #####################################

  return loss, dW;