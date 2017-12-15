from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, pdb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import utils.dgm as dgm
import utils.bnn as bnn

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

class bnn(object):
    
    def __init__(self, n_x, n_y, n_hid, y_dist, nonlinearity=tf.nn.relu, initVar=-5., batchnorm=False, wSamples=1, ckpt=None):
	self.n_x, self.n_y = n_x, n_y     # data dimensions
	self.y_dist = y_dist              # regression (gaussian) or classification 'categorical'
	self.n_hid = n_hid                # network architecture
	self.initVar = initVar            # initial variance for model parameters
	self.nonlinearity = nonlinearity  # network activation function
	self.bn = batchnorm               # type of batchnorm to use ('None', 'standard', 'bayes') 
	self.wSamps = wSamples            # number of samples from q(w)
	self.name = 'bnn'                 # model name
	self.ckpt = ckpt                  # preallocated checkpoint directory

	self.build_model()
	self.loss = self.compute_loss(self.x,self.y)
	self.sess = tf.Session()

    def train(self, Data, n_epochs, batchsize, lr, eval_samps=None, binarize=False, logging=False):
        """ Method for training the models """
	self.train_curve, self.test_curve = [],[]
        self.data_init(Data, eval_samps, batchsize)
        self.lr = self.set_learning_rate(lr)
        ## define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        
	self.train_acc, self.test_acc = np.zeros(n_epochs), np.zeros(n_epochs)
        ## initialize session and train
        epoch, step = 0, 0
        with self.sess as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if logging:
                writer = tf.summary.FileWriter(self.LOGDIR, sess.graph)

 	    while epoch < n_epochs:
                x, y = Data.next_batch(batchsize)
                if binarize == True:
                    x = self.binarize(x)
                ####
		#fd = {self.x:x, self.y:y}
                #pdb.set_trace()
                ####
                _, loss_batch = sess.run([self.optimizer, self.loss], {self.x:x, self.y:y, self.n:self.ntrain})
                if logging:
                    writer.add_summary(summary_elbo, global_step=self.global_step)

                if Data._epochs > epoch:
		    if self.y_dist=='gaussian':
		        self.train_curve.append(sess.run(self.compute_rmse(self.x, self.y), {self.x:Data.data['x_train'], self.y:Data.data['y_train'], self.n:self.ntrain}))
		        self.test_curve.append(sess.run(self.compute_rmse(self.x, self.y), {self.x:Data.data['x_test'], self.y:Data.data['y_test'], self.n:self.ntest}))
		    else:
		        self.train_curve.append(sess.run(self.compute_acc(self.x, self.y), {self.x:Data.data['x_train'], self.y:Data.data['y_train'], self.n:self.ntrain}))
		        self.test_curve.append(sess.run(self.compute_acc(self.x, self.y), {self.x:Data.data['x_test'], self.y:Data.data['y_test'], self.n:self.ntest}))

                    epoch += 1
                    saver.save(sess, self.ckpt_dir, global_step=step+1)
                    if self.y_dist == 'categorical':
                        self.print_verbose1(epoch, Data, x, y, sess)
                    elif self.y_dist == 'gaussian':
                        self.print_verbose2(epoch, Data, x, y, sess)
	    self.train_curve = np.array(self.train_curve)
	    self.test_curve = np.array(self.test_curve)
            if logging:
                writer.close()

    def build_model(self):
	self.create_placeholders()
	if self.y_dist == 'gaussian':
	    self.q = dgm.initGaussBNN(self.n_x, self.n_hid, self.n_y, 'network', initVar=self.initVar, bn=self.bn)
	    self.wTilde = dgm.sampleGaussBNN(self.q, self.n_hid)
	    self.y_m, self.y_lv = dgm.forwardPassGauss(self.x, self.wTilde, self.q, self.n_hid, self.nonlinearity, self.bn, training=True, scope='q', reuse=False)
	elif self.y_dist == 'categorical':
	    self.q = dgm.initCatBNN(self.n_x, self.n_hid, self.n_y, 'network', initVar=self.initVar, bn=self.bn)
	    self.wTilde = dgm.sampleCatBNN(self.q, self.n_hid)
	    self.y_logits = dgm.forwardPassCatLogits(self.x, self.wTilde, self.q, self.n_hid, self.nonlinearity, self.bn, training=True, scope='q', reuse=False)
	if self.y_dist=='categorical':
	    self.predictions = tf.reduce_mean(self.predict(self.x, 10, training=False),0)
	elif self.y_dist=='gaussian':
	    self.predictions = tf.reduce_mean(self.predict(self.x, 10, training=False)[0], 0)

    def compute_loss(self, x, y):
	yr = tf.tile(tf.expand_dims(y,0), [self.wSamps,1,1])
	if self.y_dist == 'gaussian':
	    ym, ylv = self.predict(x, self.wSamps, training=True)
	    self.l = -tf.reduce_sum(tf.reduce_mean(tf.square(ym-yr),axis=0)) 
	    nx = tf.cast(tf.shape(x)[0],tf.float32)
	    self.kl_term = nx*dgm.klWGaussBNN_exact(self.q, self.n_hid)/self.n
	elif self.y_dist == 'categorical':
	    y_ = self.predict(x, self.wSamps, training=True)
	    self.l = tf.reduce_sum(tf.reduce_mean(dgm.multinoulliLogDensity(yr,y_), axis=0))
	    nx = tf.cast(tf.shape(x)[0],tf.float32)
	    self.kl_term = nx*dgm.klWCatBNN_exact(self.q, self.n_hid)/self.n
	return -(self.l - self.kl_term) 
 
    def predict(self, x, n_w, training=True):
	if self.y_dist == 'gaussian':
	    return self.predictGauss(x, n_w, training)	    
	elif self.y_dist == 'categorical':
	    return tf.nn.softmax(self.predictCat(x, n_w, training))	    

    def predictGauss(self, x, n_w, training):
	self.sampleW()
	ym, ylv = self.predictConditionW(x, training)
	y_m, y_lv = tf.expand_dims(ym,0), tf.expand_dims(ylv,0)
	for sample in range(n_w-1):
	    self.sampleW()
	    ym_new, ylv_new = self.predictConditionW(x, training)
	    y_m = tf.concat([y_m, tf.expand_dims(ym_new,0)], axis=0)
	    y_lv = tf.concat([y_lv, tf.expand_dims(ym_new,0)], axis=0)
	return y_m, y_lv

    def predictCat(self, x, n_w, training):
	self.sampleW()
	y_ = self.predictConditionW(x, training)
	y_ = tf.expand_dims(y_,0)
	for sample in range(n_w-1):
	    self.sampleW()
	    y_new = self.predictConditionW(x, training)
	    y_ = tf.concat([y_, tf.expand_dims(y_new,0)], axis=0)
	return y_

    def sampleW(self):
	""" generate a sample of weights """
	if self.y_dist == 'gaussian':
	    self.wTilde = dgm.sampleGaussBNN(self.q, self.n_hid)
	elif self.y_dist == 'categorical':
	    self.wTilde = dgm.sampleCatBNN(self.q, self.n_hid)
    
    def predictConditionW(self, x, training=True):
	""" return E[p(y|x, wTilde)] (assumes wTilde~q(W) has been sampled) """
	if self.y_dist == 'gaussian':
	    return dgm.forwardPassGauss(x, self.wTilde, self.q, self.n_hid, self.nonlinearity, self.bn, training, scope='q')
	elif self.y_dist == 'categorical':
	    return dgm.forwardPassCatLogits(x, self.wTilde, self.q, self.n_hid, self.nonlinearity, self.bn, training, scope='q')
    def binarize(self, x):
	""" sample values from a bernoulli distribution """
	return np.random.binomial(1,x)

    def predict_new(self, x):
        saver = tf.train.Saver()
        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            saver.restore(session, ckpt.model_checkpoint_path)
            preds = session.run(self.predictions, {self.x:x})
        return preds

    def compute_acc(self, x, y):
	y_ = tf.reduce_mean(self.predict(x, self.wSamps, training=False), axis=0)
	return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, axis=1), tf.argmax(y,axis=1)), tf.float32))

    def compute_rmse(self, x, y):
	ym, ylv = self.predict(x, self.wSamps, training=False)
	ym, ylv = tf.reduce_mean(ym, 0), tf.reduce_mean(ylv, 0)
	return tf.sqrt(tf.reduce_mean(tf.square(y-ym)))

    def create_placeholders(self):
	self.x_train = tf.placeholder(tf.float32, shape=[None, self.n_x], name='x_train')
	self.y_train = tf.placeholder(tf.float32, shape=[None, self.n_y], name='y_train')
	self.x = tf.placeholder(tf.float32, shape=[None, self.n_x], name='x')
	self.y = tf.placeholder(tf.float32, shape=[None, self.n_y], name='y')
	self.n = tf.placeholder(tf.float32, shape=[], name='n')
	
    def data_init(self, data, eval_samps, bs):
	self.N = data.N
	self.ntrain = data.TRAIN_SIZE
	self.ntest = data.TEST_SIZE
	if eval_samps == None:
            self.eval_samps = self.ntrain     # evaluation training set size
            self.eval_samps_test = self.ntest # evaluation test set size
        else:
            self.eval_samps_train = eval_samps
            self.eval_samps_test = eval_samps
        self.data_name = data.NAME             # dataset being used   
        self._allocate_directory()             # logging directory

    def _allocate_directory(self):
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

    def print_verbose1(self, epoch, data, x, y, sess):
	""" verbose for classification problems """
	acc_train = sess.run(self.compute_acc(self.x, self.y), {self.x:data.data['x_train'], self.y:data.data['y_train'], self.n:self.ntrain})
	acc_test = sess.run(self.compute_acc(self.x, self.y), {self.x:data.data['x_test'], self.y:data.data['y_test'], self.n:self.ntest})
	kl, ll = sess.run([self.kl_term, self.l], {self.x:x, self.y:y, self.n:x.shape[0]})
	self.train_acc[epoch-1], self.test_acc[epoch-1] = acc_train, acc_test
	print('Epoch {}: Training: {:5.3f}, Testing: {:5.3f}, KL: {:5.3f}, Data: {:5.3f}'.format(epoch, acc_train, acc_test, kl, ll))

    def print_verbose2(self, epoch, data, x, y, sess):
	""" verbose for regression problems """
	rmse_train = sess.run(self.compute_rmse(self.x, self.y), {self.x:data.data['x_train'], self.y:data.data['y_train'], self.n:self.ntrain})
	rmse_test = sess.run(self.compute_rmse(self.x, self.y), {self.x:data.data['x_test'], self.y:data.data['y_test'], self.n:self.ntest})
	kl, ll = sess.run([self.kl_term, self.l], {self.x:x, self.y:y, self.n:self.ntrain})
	self.train_acc[epoch-1], self.test_acc[epoch-1] = rmse_train, rmse_test
	print('Epoch {}: Training: {:5.3f}, Testing: {:5.3f}, KL: {:5.3f}, Data: {:5.3f}'.format(epoch, rmse_train, rmse_test, kl, ll))
