from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import pdb

""" Generate a data structure to support SSL models. Expects:
x - np array: N rows, d columns
y - np array: N rows, k columns (one-hot encoding)
"""


class Container:
    """ Class for appropriate data structures """
    def __init__(self, x, y, x_test=None, y_test=None, train_proportion=0.9, dataset='moons', seed=None):
	
	self.INPUT_DIM = x.shape[1]
	self.NUM_CLASSES = y.shape[1]
	self.NAME = dataset
	
        if x_test is None:
	    self.N = x.shape[0]
	    self.TRAIN_SIZE = int(np.round(train_proportion * self.N))
	    self.TEST_SIZE = int(self.N-self.TRAIN_SIZE)
	else:
	    self.TRAIN_SIZE = x.shape[0]
	    self.TEST_SIZE = x_test.shape[0]
            self.N = self.TRAIN_SIZE + self.TEST_SIZE

	# create necessary data splits
	if seed:
	    np.random.seed(seed)
    	if x_test is None:
	    xtrain, ytrain, xtest, ytest = self._split_data(x,y)
	else:
	    xtrain, ytrain, xtest, ytest = x, y, x_test, y_test

	# create appropriate data dictionaries
	self.data = {}
	self.data['x_train'], self.data['y_train'] = xtrain, ytrain
	self.data['x_test'], self.data['y_test'] = xtest, ytest

	# counters and indices for minibatching
	self._start = 0
	self._epochs = 0

    def _split_data(self, x, y):
	""" split the data according to the proportions """
	indices = range(self.N)
	np.random.shuffle(indices)
	train_idx, test_idx = indices[:self.TRAIN_SIZE], indices[self.TRAIN_SIZE:]
	return (x[train_idx,:], y[train_idx,:], x[test_idx,:], y[test_idx,:])

    def next_batch(self, batch_size, shuffle=True):
    	"""Return the next `batch_size` examples from this data set."""
        start = self._start
    	# Shuffle for the first epoch
    	if self._epochs == 0 and start == 0 and shuffle:
      	    perm0 = np.arange(self.TRAIN_SIZE)
	    np.random.shuffle(perm0)
      	    self.data['x_train'], self.data['y_train'] = self.data['x_train'][perm0,:], self.data['y_train'][perm0,:]
   	# Go to the next epoch
    	if start + batch_size > self.TRAIN_SIZE:
      	    # Finished epoch
      	    self._epochs += 1
      	    # Get the rest examples in this epoch
      	    rest_num_examples = self.TRAIN_SIZE - start
      	    inputs_rest_part = self.data['x_train'][start:self.TRAIN_SIZE,:]
      	    labels_rest_part = self.data['y_train'][start:self.TRAIN_SIZE,:]
      	    # Shuffle the data
      	    if shuffle:
        	perm = np.arange(self.TRAIN_SIZE)
		np.random.shuffle(perm)
        	self.data['x_train'] = self.data['x_train'][perm]
        	self.data['y_train'] = self.data['y_train'][perm]
      	    # Start next epoch
      	    start = 0
      	    self._start = batch_size - rest_num_examples
      	    end = self._start
      	    inputs_new_part = self.data['x_train'][start:end,:]
      	    labels_new_part = self.data['y_train'][start:end,:]
      	    return np.concatenate((inputs_rest_part, inputs_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    	else:
      	    self._start += batch_size
      	    end = self._start
	    return self.data['x_train'][start:end,:], self.data['y_train'][start:end,:]
  
    def next_batch_shuffle(self, batch_size):
	""" Return a random subsample of the data """
	perm0 = np.arange(self.TRAIN_SIZE)
        np.random.shuffle(perm0)
        self.data['x_train'], self.data['y_train'] = self.data['x_train'][perm0,:], self.data['y_train'][perm0,:]
	return self.data['x_train'][:batch_size], self.data['y_train'][:batch_size]

    def sample_train(self, n_samples=1000):
	perm_train = np.arange(self.TRAIN_SIZE)
	np.random.shuffle(perm_train)
        self.data['x_train'], self.data['y_train'] = self.data['x_train'][perm_train,:], self.data['y_train'][perm_train,:]
	return self.data['x_train'][:n_samples], self.data['y_train'][:n_samples]
	
    def sample_test(self, n_samples=1000):
	perm_test = np.arange(self.TEST_SIZE)
	np.random.shuffle(perm_test)
        self.data['x_test'], self.data['y_test'] = self.data['x_test'][perm_test,:], self.data['y_test'][perm_test,:]
	return self.data['x_test'][:n_samples], self.data['y_test'][:n_samples]

    def reset_counters(self):
	# counters and indices for minibatching
	self._start_labeled, self._start_unlabeled = 0, 0
	self._epochs_labeled = 0
	self._epochs_unlabeled = 0
        self._start_regular = 0
        self._epochs_regular = 0


