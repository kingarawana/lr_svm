import numpy as np

def svm_loss_naive(W, X, y, reg):
  loss = 0.0
  dW = np.zeros(W.shape)

  #####################################
  # Your Code Here

  # compute the loss and the gradient

  num_classes = W.shape[0]
  num_images = len(y)

  for i in range(num_images):
    scores = W.dot(X[:, i]) # each row is a class score
    correct_score = scores[y[i]]
    for j in range(num_classes):
      if(j != y[i]):
        margin = scores[j] - correct_score + 1
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

def svm_loss_vectorized(W, X, y, reg):
  loss = 0.0
  dW = np.zeros(W.shape)

  #####################################
  # Your Code Here



  num_classes = W.shape[0]
  num_images = X.shape[1]

  scores = W.dot(X)
  correct_scores = scores[y, range(num_images)]
  margins = scores - correct_scores + 1
  margins[y, range(num_images)] = 0
  margins = np.maximum(np.zeros((num_classes, num_images)), margins)

  loss = np.sum(margins)
  loss += 0.5 * reg * np.sum(W * W)

  binary = margins
  binary[binary > 0] = 1
  column_count = np.sum(binary, axis=0)
  binary[y, range(num_images)] = -column_count
  # print binary[1:20, 1:20]
  dW = np.dot(binary, X.T)

  loss /= num_images
  dW /= num_images

  dW += reg * W




  

  #####################################

  return loss, dW;