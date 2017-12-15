from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib.distributions import RelaxedOneHotCategorical as Gumbel
import pdb

""" Module containing shared functions and structures for DGMS """

glorotNormal = xavier_initializer(uniform=False)
initNormal = tf.random_normal_initializer(stddev=1e-3)

############# Probability functions ##############

def gaussianLogDensity(inputs, mu, log_var):
    """ Gaussian log density """
    b_size = tf.cast(tf.shape(mu)[0], tf.float32)
    D = tf.cast(tf.shape(inputs)[-1], tf.float32)
    xc = inputs - mu
    return -0.5*(tf.reduce_sum((xc * xc) / tf.exp(log_var), axis=-1) + tf.reduce_sum(log_var, axis=-1) + D * tf.log(2.0*np.pi))

def standardNormalLogDensity(inputs):
    """ Standard normal log density """
    mu = tf.zeros_like(inputs)
    log_var = tf.log(tf.ones_like(inputs))
    return gaussianLogDensity(inputs, mu, log_var)

def multinoulliLogDensity(inputs, logits):
    """ Categorical log density """
    return -tf.nn.softmax_cross_entropy_with_logits(labels=inputs, logits=logits)

def multinoulliUniformLogDensity(inputs):
    """ Uniform Categorical log density """
    logits = tf.ones_like(inputs)
    return -tf.nn.softmax_cross_entropy_with_logits(labels=inputs, logits=logits)

def sampleNormal(mu, logvar, mc_samps=1):
    """ return a reparameterized sample from a Gaussian distribution """
    shape = tf.concat([tf.constant([mc_samps]), tf.shape(mu)], axis=-1)
    eps = tf.random_normal(shape, dtype=tf.float32)
    return mu + eps * tf.sqrt(tf.exp(logvar))

def standardNormalKL(mu, logvar):
    """ compute the KL divergence between a Gaussian and standard normal """
    return -0.5 * tf.reduce_sum(1 + logvar - mu**2 - tf.exp(logvar), axis=-1)

############## Neural Network modules ##############

def initNetwork(n_in, n_hid, n_out, vname):
    weights = {}
    for layer, neurons in enumerate(n_hid):
        weight_name, bias_name = 'W'+str(layer), 'b'+str(layer)
        if layer == 0:
       	    weights[weight_name] = tf.get_variable(shape=[n_in, n_hid[layer]], name=vname+weight_name, initializer=glorotNormal)
    	else:
    	    weights[weight_name] = tf.get_variable(shape=[n_hid[layer-1], n_hid[layer]], name=vname+weight_name, initializer=glorotNormal)
    	weights[bias_name] = tf.get_variable(shape=[n_hid[layer]], name=vname+bias_name, initializer=initNormal)
    return weights

def initGaussNet(n_in, n_hid, n_out, vname):
    """ Initialize the weights of a network parameterizeing a Gaussian distribution"""
    weights = initNetwork(n_in, n_hid, n_out, vname)
    weights['Wmean'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wmean', initializer=initNormal)
    weights['bmean'] = tf.get_variable(shape=[n_out], name=vname+'bmean', initializer=initNormal)
    weights['Wvar'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wvar', initializer=initNormal)
    weights['bvar'] = tf.get_variable(shape=[n_out], name=vname+'bvar', initializer=initNormal)
    return weights

def initCatNet(n_in, n_hid, n_out, vname):
    """ Initialize the weights of a network parameterizeing a Gaussian distribution"""
    weights = initNetwork(n_in, n_hid, n_out, vname)
    weights['Wout'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wout', initializer=initNormal)
    weights['bout'] = tf.get_variable(shape=[n_out], name=vname+'bout', initializer=initNormal)
    return weights

def forwardPass(x, weights, q, n_h, nonlinearity, bn, training, scope, reuse):
    h = x
    for layer, neurons in enumerate(n_h):
	weight_name, bias_name = 'W'+str(layer), 'b'+str(layer)
	weight_mean, bias_mean = 'W'+str(layer)+'_mean', 'b'+str(layer)+'_mean'
	hq = tf.matmul(h, q[weight_mean]) + q[bias_mean]
	h = tf.matmul(h, weights[weight_name]) + weights[bias_name]
	if bn == 'standard':
	    name = scope+'_bn'+str(layer)
	    h = tf.layers.batch_normalization(h, training=training, name=name, reuse=reuse)
	elif bn == 'bayes':
	    h = bayesBatchNorm(h, hq, q, layer, training=training)
	h = nonlinearity(h)
    return h	

def forwardPassGauss(x, weights, q, n_h, nonlinearity, bn, training=True, scope='scope', reuse=True):
    """ Forward pass through the network with given weights - Gaussian output """
    h = forwardPass(x, weights, q, n_h, nonlinearity, bn, training, scope, reuse)
    mean = tf.matmul(h, weights['Wmean']) + weights['bmean']
    logVar = tf.matmul(h, weights['Wvar']) + weights['bvar']
    return mean, logVar

def forwardPassCatLogits(x, weights, q, n_h, nonlinearity, bn, training=True, scope='scope', reuse=True):
    """ Forward pass through the network with weights as a dictionary """
    h = forwardPass(x, weights, q, n_h, nonlinearity, bn, training, scope, reuse) 
    logits = tf.matmul(h, weights['Wout']) + weights['bout']
    return logits

def forwardPassCat(x, weights, q, n_h, nonlinearity, bn=False, training=True, scope='scope', reuse=True):
    """ Forward pass through network with given weights - Categorical output """
    return tf.nn.softmax(forwardPassCatLogits(x, weights, q, n_h, nonlinearity, bn, training, scope, reuse))

