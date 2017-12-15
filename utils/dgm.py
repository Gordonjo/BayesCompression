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

############## Bayesian Neural Network modules ############## 

def initBNN(n_in, n_hid, n_out, initVar, vname, bn=None):
    """ initialize the weights that define a general Bayesian neural network """
    weights = {}
    for layer, neurons in enumerate(n_hid):
        weight_mean, bias_mean = 'W'+str(layer)+'_mean', 'b'+str(layer)+'_mean'
        weight_logvar, bias_logvar = 'W'+str(layer)+'_logvar', 'b'+str(layer)+'_logvar'
        if layer == 0:
            weights[weight_mean] = tf.get_variable(shape=[n_in, n_hid[layer]], name=vname+weight_mean, initializer=xavier_initializer())
            weights[weight_logvar] = tf.Variable(tf.fill([n_in,n_hid[layer]], initVar), name=vname+weight_logvar)
        else:
            weights[weight_mean] = tf.get_variable(shape=[n_hid[layer-1], n_hid[layer]],name=vname+weight_mean, initializer=xavier_initializer())
            weights[weight_logvar] = tf.Variable(tf.fill([n_hid[layer-1], n_hid[layer]], initVar), name=vname+weight_logvar)
        weights[bias_mean] = tf.Variable(tf.zeros([n_hid[layer]]) + 1e-1, name=vname+bias_mean)
        weights[bias_logvar] = tf.Variable(tf.fill([n_hid[layer]], initVar), name=vname+bias_logvar)
	if bn=='bayes':	
	    scale, beta, mean, var = 'scale'+str(layer), 'beta'+str(layer), 'mean'+str(layer), 'var'+str(layer)
            weights[scale] = tf.Variable(tf.ones(n_hid[layer]), name=vname+scale)
            weights[beta] = tf.Variable(tf.zeros(n_hid[layer]), name=vname+beta)
            weights[mean] = tf.Variable(tf.zeros(n_hid[layer]), name=vname+mean, trainable=False)
            weights[var] = tf.Variable(tf.ones(n_hid[layer]), name=vname+var, trainable=False)
    return weights    

def initCatBNN(n_in, n_hid, n_out, vname, initVar=-5, bn=None):
    """ initialize a BNN with categorical output (classification """
    weights = initBNN(n_in, n_hid, n_out, initVar, vname, bn)
    weights['Wout_mean'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wout_mean', initializer=xavier_initializer())
    weights['Wout_logvar'] = tf.Variable(tf.fill([n_hid[-1], n_out], initVar), name=vname+'Wout_logvar')
    weights['bout_mean'] = tf.Variable(tf.zeros([n_out]) + 1e-1, name=vname+'bout_mean')
    weights['bout_logvar'] = tf.Variable(tf.fill([n_out], value=initVar), name=vname+'bout_logvar')
    return weights

def initGaussBNN(n_in, n_hid, n_out, vname, initVar=-5, bn=None):
    """ TODO: initialize a BNN with Gaussian output (regression) """
    weights = initBNN(n_in, n_hid, n_out, initVar, vname, bn)
    weights['Wmean_mean'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wmean_mean', initializer=xavier_initializer())
    weights['Wmean_logvar'] = tf.Variable(tf.fill([n_hid[-1], n_out], initVar), name=vname+'Wmean_logvar')
    weights['bmean_mean'] = tf.Variable(tf.zeros([n_out]) + 1e-1, name=vname+'bmean_mean')
    weights['bmean_logvar'] = tf.Variable(tf.fill([n_out], value=initVar), name=vname+'bmean_logvar')
    weights['Wvar_mean'] = tf.get_variable(shape=[n_hid[-1], n_out], name=vname+'Wvar_mean', initializer=xavier_initializer())
    weights['Wvar_logvar'] = tf.Variable(tf.fill([n_hid[-1], n_out], initVar), name=vname+'Wvar_logvar')
    weights['bvar_mean'] = tf.Variable(tf.zeros([n_out]) + 1e-1, name=vname+'bvar_mean')
    weights['bvar_logvar'] = tf.Variable(tf.fill([n_out], value=initVar), name=vname+'bvar_logvar')
    return weights

def sampleBNN(weights, n_hid):
    """ sample weights from a variational approximation """
    wTilde = {}
    for layer in range(len(n_hid)):
	wName, bName = 'W'+str(layer), 'b'+str(layer)
	meanW, meanB = weights['W'+str(layer)+'_mean'], weights['b'+str(layer)+'_mean']
	logvarW, logvarB = weights['W'+str(layer)+'_logvar'], weights['b'+str(layer)+'_logvar']
	wTilde[wName], wTilde[bName] = tf.squeeze(sampleNormal(meanW, logvarW)), tf.squeeze(sampleNormal(meanB, logvarB))
    return wTilde

def sampleCatBNN(weights, n_hid):
    """ return a sample from weights of a categorical BNN """
    wTilde = sampleBNN(weights, n_hid)
    meanW, meanB = weights['Wout_mean'], weights['bout_mean']
    logvarW, logvarB = weights['Wout_logvar'], weights['bout_logvar']
    wTilde['Wout'], wTilde['bout'] = tf.squeeze(sampleNormal(meanW, logvarW)), tf.squeeze(sampleNormal(meanB, logvarB))
    return wTilde

def sampleGaussBNN(weights, n_hid):
    """ return a sample from weights of a Gaussian BNN """
    wTilde = sampleBNN(weights, n_hid)
    meanW, meanB = weights['Wmean_mean'], weights['bmean_mean']
    logvarW, logvarB = weights['Wmean_logvar'], weights['bmean_logvar']
    wTilde['Wmean'], wTilde['bmean'] = tf.squeeze(sampleNormal(meanW, logvarW),0), tf.squeeze(sampleNormal(meanB, logvarB),0)
    meanW, meanB = weights['Wvar_mean'], weights['bvar_mean']
    logvarW, logvarB = weights['Wvar_logvar'], weights['bvar_logvar']
    wTilde['Wvar'], wTilde['bvar'] = tf.squeeze(sampleNormal(meanW, logvarW),0), tf.squeeze(sampleNormal(meanB, logvarB),0)
    return wTilde

def klWBNN(q, W, n_hid, dist):
    """ estimate KL(q(w)||p(w)) as logp(w) - logq(w) 
	currently only p(w) = N(w;0,1) implemented """
    l_pw, l_qw = 0,0
    for layer, neurons in enumerate(n_hid):
        w, b =  W['W'+str(layer)], W['b'+str(layer)]
	wMean, bMean = q['W'+str(layer)+'_mean'], q['b'+str(layer)+'_mean']
        wLv, bLv = q['W'+str(layer)+'_logvar'], q['b'+str(layer)+'_logvar']
	l_pw += tf.reduce_sum(standardNormalLogDensity(w)) + tf.reduce_sum(standardNormalLogDensity(b))
	l_qw += tf.reduce_sum(gaussianLogDensity(w,wMean,wLv)) + tf.reduce_sum(gaussianLogDensity(b,bMean,bLv))
    return l_pw, l_qw

def klBNN_exact(q, n_hid):
    """ compute exact KL(q||N(0,1)) """
    kl = 0
    for layer, neurons in enumerate(n_hid):
	wMean, bMean = q['W'+str(layer)+'_mean'], tf.expand_dims(q['b'+str(layer)+'_mean'],1)
        wLv, bLv = q['W'+str(layer)+'_logvar'], tf.expand_dims(q['b'+str(layer)+'_logvar'],1)
	kl += tf.reduce_sum(standardNormalKL(wMean, wLv)) + tf.reduce_sum(standardNormalKL(bMean, bLv))
    return kl 
    
def klWCatBNN(q, W, n_hid, dist='Gaussian'):
    """ estimate KL(q||p) as logp(w) - logq(w) for a categorical BNN """
    l_pw, l_qw = klWBNN(q, W, n_hid, dist)
    w, b = W['Wout'], W['bout']
    wMean, bMean, wLv, bLv = q['Wout_mean'], q['bout_mean'], q['Wout_logvar'], q['bout_logvar']
    l_pw += tf.reduce_sum(standardNormalLogDensity(w)) + tf.reduce_sum(standardNormalLogDensity(b))
    l_qw += tf.reduce_sum(gaussianLogDensity(w,wMean,wLv)) + tf.reduce_sum(gaussianLogDensity(b,bMean,bLv))
    return l_pw - l_qw

def klWCatBNN_exact(q, n_hid):
    """ compute exact KL(q||p) with standard normal p(w) for a categorical BNN """
    kl = klBNN_exact(q, n_hid)
    wMean, bMean = q['Wout_mean'], tf.expand_dims(q['bout_mean'],1)
    wLv, bLv = q['Wout_logvar'], tf.expand_dims(q['bout_logvar'],1)
    kl += tf.reduce_sum(standardNormalKL(wMean, wLv)) + tf.reduce_sum(standardNormalKL(bMean, bLv))
    return kl

def klWGaussBNN_exact(q, n_hid):
    """ compute exact KL(q||p) with standard normal p(w) for a categorical BNN """
    kl = klBNN_exact(q, n_hid)
    wMean, bMean = q['Wmean_mean'], tf.expand_dims(q['bmean_mean'],1)
    wLv, bLv = q['Wmean_logvar'], tf.expand_dims(q['bmean_logvar'],1)
    kl += tf.reduce_sum(standardNormalKL(wMean, wLv)) + tf.reduce_sum(standardNormalKL(bMean, bLv))
    wMean, bMean = q['Wvar_mean'], tf.expand_dims(q['bvar_mean'],1)
    wLv, bLv = q['Wvar_logvar'], tf.expand_dims(q['bvar_logvar'],1)
    kl += tf.reduce_sum(standardNormalKL(wMean, wLv)) + tf.reduce_sum(standardNormalKL(bMean, bLv))
    return kl

def averageVarBNN(q, n_hid):
    """ return the average (log) variance of variational distribution """
    totalVar, numParams = 0,0
    for layer in range(len(n_hid)):
        variances = tf.reshape(q['W'+str(layer)+'_logvar'], [-1])
        totalVar += tf.reduce_sum(tf.exp(variances))
        numParams += tf.cast(tf.shape(variances)[0], dtype=tf.float32)
        variances = tf.reshape(q['b'+str(layer)+'_logvar'], [-1])
        totalVar += tf.reduce_sum(tf.exp(variances))
        numParams += tf.cast(tf.shape(variances)[0], dtype=tf.float32)
    variances = tf.reshape(q['Wout_logvar'], [-1])
    totalVar += tf.reduce_sum(tf.exp(variances))
    numParams += tf.cast(tf.shape(variances)[0], tf.float32)
    variances = tf.reshape(q['bout_logvar'], [-1])
    totalVar += tf.reduce_sum(tf.exp(variances))
    numParams += tf.cast(tf.shape(variances)[0], dtype=tf.float32)    
    return totalVar/numParams 	
 
def bayesBatchNorm(inputs, var_inputs, q, layer, training, decay=0.9, epsilon=1e-3):
    """ BatchNorm for BNNs """
    layer = str(layer)
    if training==True:
        batch_mean, batch_var = tf.nn.moments(var_inputs,[0])
        train_mean = tf.assign(q['mean'+layer],
                               q['mean'+layer] * decay + batch_mean * (1 - decay))
        train_var = tf.assign(q['var'+layer],
                              q['var'+layer] * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, q['beta'+layer], q['scale'+layer], epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            q['mean'+layer], q['var'+layer], q['beta'+layer], q['scale'+layer], epsilon)
