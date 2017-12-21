from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, pdb
import matplotlib.pyplot as plt
import numpy as np
from utils.dgm import *

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

""" Base class for simple deep neural nets """

class NeuralNet(object):

    def __init__(self, n_in, n_hid, n_out, y_dist, nonlinearity=tf.nn.relu, batchnorm=False, name='nnet', ckpt=None):

	self.n_in, self.n_hid = n_in, n_hid,          # network input and architecture
	self.n_out, self.y_dist = n_out, y_dist       # number of classes and output distribution 
	self.nonlinearity = nonlinearity              # network activation function
	self.batchnorm = batchnorm                    # binary flag for batch normalization
	self.ckpt = ckpt                              # checkpoint directory
	self.name = name                              # network name 
	
	self.create_placeholders()	
	if y_dist=='gaussian':
	    self.weights = initGaussNet(self.n_in, self.n_hid, self.n_out, self.name)
	elif y_dist=='categorical':
	    self.weights = initCatNet(self.n_in, self.n_hid, self.n_out, self.name)
	    self.predictions = self.forwardPassCat(self.x)
 	self.loss = self.computeLoss(self.x, self.y)
	self.sess = tf.Session()	

    def forwardPass(self, inputs, training, reuse):
	h = inputs
        for layer, neurons in enumerate(self.n_hid):
            weight_name, bias_name = 'W'+str(layer), 'b'+str(layer)
            h = tf.matmul(h, self.weights[weight_name]) + self.weights[bias_name]
            if self.batchnorm:
                name = self.name+'_bn'+str(layer)
                h = tf.layers.batch_normalization(h, training=training, name=name, reuse=reuse, momentum=0.99)
            h = self.nonlinearity(h)
        return h

    def forwardPassGauss(self, inputs, training=True, reuse=True):
	""" Forward pass through the network with given weights - Gaussian output """
	h = self.forwardPass(inputs, training, reuse)
	mean = tf.matmul(h, self.weights['Wmean']) + self.weights['bmean']
	logVar = tf.matmul(h, self.weights['Wvar']) + self.weights['bvar']
	return mean, logVar

    def samplePassGauss(self, inputs, training=True, reuse=True):
        """ Forward pass through the network with given weights - Gaussian sampling """
        mean, logVar = self.forwardPassGauss(inputs, training, reuse)
        return mean, logVar, sampleNormal(mean, logVar, mc_samps)
    
    def forwardPassCatLogits(self, inputs, training=True, reuse=True):
        """ Forward pass through the network with weights as a dictionary """
        h = self.forwardPass(inputs, training, reuse)
        logits = tf.matmul(h, self.weights['Wout']) + self.weights['bout']
        return logits
    
    def forwardPassCat(self, inputs, training=True, reuse=True):
        """ Forward pass through network with given weights - Categorical output """
        return tf.nn.softmax(self.forwardPassCatLogits(inputs, training, reuse))
    
    def forwardPassBernoulli(self, inputs, training=True, reuse=True):
        """ Forward pass through the network with given weights - Bernoulli output """
        return tf.nn.sigmoid(self.forwardPassCatLogits(inputs, training, reuse))

    def computeLoss(self, x, y):
	if self.y_dist=='gaussian':
	    pass
	if self.y_dist=='categorical':
	    y_ = self.forwardPassCatLogits(x)
	    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))

    def fit(self, Data, params):
	""" fit the model with data """
        self.allocate_dir(Data)
        self.lr = self.set_learning_rate(params['lr'])
        ## define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        var_list = [V for V in tf.trainable_variables() if self.name in V.name]
        self.optimizer = optimizer.minimize(self.loss, var_list=var_list)

        epoch, step = 0, 0
        with self.sess as sess:
            sess.run(tf.global_variables_initializer())
            # instantiate different savers for student, teacher and restore teacher
            saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
            while epoch < params['n_epochs']:
                x, y = Data.next_batch(params['batchsize'])
                if params['binarize'] == True:
                    x = self.Teacher.binarize(x)
                _, loss_batch = sess.run([self.optimizer, self.loss], {self.x:x, self.y:y})
                if Data._epochs > epoch:
		    epoch += 1
                    saver.save(sess, self.ckpt_dir, global_step=epoch)
                    self.print_verbose(Data, x, y, sess, epoch)

    def create_placeholders(self):
        self.x_train = tf.placeholder(tf.float32, shape=[None, self.n_in], name='x_train')
        self.y_train = tf.placeholder(tf.float32, shape=[None, self.n_out], name='y_train')
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_in], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_out], name='y')
        self.n = tf.placeholder(tf.float32, shape=[], name='n')

    def print_verbose(self, Data, x, y, sess, epoch):
        loss = sess.run(self.computeLoss(self.x, self.y), {self.x:x, self.y:y})
        print('Epoch: {}, Loss: {:5.3f}'.format(epoch, loss))

    def allocate_dir(self, Data):
        self.data_name = Data.NAME
        if self.ckpt == None:
            self.LOGDIR = './graphs/'+self.name+'-'+self.data_name+'/'
            self.ckpt_dir = './ckpt/'+self.name+'-'+self.data_name+'/'
        else:
            self.LOGDIR = 'graphs/'+self.ckpt+'/'
            self.ckpt_dir = './ckpt/' + self.ckpt + '/'
        if not os.path.isdir(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        if not os.path.isdir(self.LOGDIR):
            os.mkdir(self.LOGDIR)

    def set_learning_rate(self, lr):
        """ Set learning rate """
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        if len(lr) == 1:
            return lr[0]
        else:
            start_lr, rate, final_lr = lr
            return tf.train.polynomial_decay(start_lr, self.global_step, rate, end_learning_rate=final_lr)

    def predict_new(self, x):
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
            preds = session.run(self.predictions, {self.x:x})
        return preds
