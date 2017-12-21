from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, pdb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils.dgm import *

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

class Student(object):

    def __init__(self, Teacher, n_x, n_y, n_hid, y_dist, nonlinearity=tf.nn.relu, batchnorm=False, ckpt=None):
	self.Teacher = Teacher            # BNN to learn from / compress
        self.n_x, self.n_y = n_x, n_y     # data dimensions
        self.y_dist = y_dist              # regression (gaussian) or classification 'categorical'
        self.n_hid = n_hid                # network architecture
        self.nonlinearity = nonlinearity  # network activation function
        self.bn = batchnorm               # type of batchnorm to use ('None', 'standard', 'bayes') 
        self.name = 'student'             # model name
        self.ckpt = ckpt                  # preallocated checkpoint directory

        self.build_model()
        self.compute_loss(self.x)
        self.sess = tf.Session()

    def train(self, Data, params):
	self.allocate_dir(Data)
        self.lr = self.set_learning_rate(params['lr'])
	saver = tf.train.Saver()
        ## define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
	var_list = [V for V in tf.trainable_variables() if self.name in V.name]
	self.optimizer = optimizer.minimize(self.loss, var_list=var_list)

        epoch, step = 0, 0
        with self.sess as sess:
	    ### load teacher variables from Teacher.ckpt_dir
            sess.run(tf.global_variables_initializer())
	    ### make sure teacher variables were not initialized above
            saver = tf.train.Saver()
            while epoch < params['n_epochs']:
                x, y = Data.next_batch(params['batchsize'])
                if params['binarize'] == True:
                    x = self.Teacher.binarize(x)
                ####
                #fd = {self.x:x, self.y:y}
                #pdb.set_trace()
                ####
                _, loss_batch = sess.run([self.optimizer, self.loss], {self.x:x})
                if Data._epochs > epoch:
                    epoch += 1
		    saver.save(sess, self.ckpt_dir, global_step=epoch)
                    self.print_verbose(Data, x, sess, epoch)

    def build_model(self):
        self.create_placeholders()
        if self.y_dist == 'gaussian':
            self.q = initGaussNet(self.n_x, self.n_hid, self.n_y, 'student')
            self.y_m, self.y_lv = forwardPassGauss(self.x, self.q, self.n_hid, self.nonlinearity, self.bn, training=True, scope='student', reuse=False)
            self.predictions = self.predict(self.x, training=False)
        elif self.y_dist == 'categorical':
            self.q = initCatNet(self.n_x, self.n_hid, self.n_y, 'student')
            self.y_logits = forwardPassCatLogits(self.x, self.q, self.n_hid, self.nonlinearity, self.bn, training=True, scope='student', reuse=False)
            self.predictions = self.predict(self.x)

    def compute_loss(self, x):
	if self.y_dist=='gaussian':
	    pass
	else:
	    y_t = tf.reduce_mean(self.Teacher.predict(x, 50, False),0) 
	    y_s = self.predict(x, predType='logits')
	    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_t, logits=y_s))

    def compute_prior(self):
	""" Any priors over the weights of q """
	pass

    def predict(self, x, training=True, predType='probs'):    
	if self.y_dist=='gaussian':
	    pass
	else:
	    if predType=='logits':
	        return forwardPassCatLogits(x, self.q, self.n_hid, self.nonlinearity, self.bn, scope='student')
	    else:
		return forwardPassCat(x, self.q, self.n_hid, self.nonlinearity, self.bn, scope='student')

    def set_learning_rate(self, lr):
        """ Set learning rate """
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        if len(lr) == 1:
            return lr[0]
        else:
            start_lr, rate, final_lr = lr
            return tf.train.polynomial_decay(start_lr, self.global_step, rate, end_learning_rate=final_lr)
 
    def create_placeholders(self):
	self.x_train = tf.placeholder(tf.float32, shape=[None, self.n_x], name='x_train')
        self.y_train = tf.placeholder(tf.float32, shape=[None, self.n_y], name='y_train')
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_x], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_y], name='y')
        self.n = tf.placeholder(tf.float32, shape=[], name='n')

    def print_verbose(self, Data, x, sess, epoch):
	loss = sess.run(self.loss, {self.x:x})
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

    def predict_new(self, x):
        saver = tf.train.Saver()
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
            preds = session.run(self.predictions, {self.x:x})
        return preds

