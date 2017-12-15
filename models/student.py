from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, pdb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import utils.dgm as dgm

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

class student(object):

     def __init__(self, n_x, n_y, n_hid, y_dist, nonlinearity, batchnorm, ckpt=None):
	self.n_x, self.n_y = n_x, n_y     # data dimensions
        self.y_dist = y_dist              # regression (gaussian) or classification 'categorical'
        self.n_hid = n_hid                # network architecture
        self.nonlinearity = nonlinearity  # network activation function
        self.bn = batchnorm               # type of batchnorm to use ('None', 'standard', 'bayes') 
        self.name = 'student'             # model name
        self.ckpt = ckpt                  # preallocated checkpoint directory

	self.build_model()
	self.compute_loss()
	self.sess = tf.Session()

     def compress(self, Data, params, teacher):
	pass

     def build_model(self):
	pass

     def compute_loss(self):
	pass






