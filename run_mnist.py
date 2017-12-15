import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle, sys, pdb, gzip, cPickle
import numpy as np
from sklearn.metrics import log_loss
import tensorflow as tf
from data.Container import Container as Data
from data.mnist import mnist 
from models.bnn import bnn
### Script to run a BNN experiment over the moons data

# argv[1] - bn type ('None', 'standard', 'bayes')
# argv[2] - number of runs (max 10)

bn_type, runs = sys.argv[1], int(sys.argv[2])
seeds = np.array([111,222,333,444,555,666,777,888,999,101010])
threshold = 0.1
mnist_data = mnist(threshold=threshold)

## Specify model parameters
lr = (3e-4,)
n_hidden = [256,256]
n_epochs, batchsize = 5, 1024 
initVar, eval_samps = -10.0, None

train_none, train_bn, train_bbn = [],[],[]
test_none, test_bn, test_bbn = [],[],[]

batchnorm = 'None' 
for run in range(runs):
    print("Starting work on run: {}".format(run))
    seed = seeds[run]
    np.random.seed(seed)
    ## load data 
    data = Data(mnist_data.x_train, mnist_data.y_train, x_test=mnist_data.x_test, y_test=mnist_data.y_test, dataset='mnist', seed=seed)
    n_x, n_y = data.INPUT_DIM, data.NUM_CLASSES
    
    model = bnn(n_x, n_y, n_hidden, 'categorical', initVar=initVar, batchnorm=batchnorm, wSamples=1)
    model.train(data, n_epochs, batchsize, lr, eval_samps=eval_samps, binarize=True)

    pdb.set_trace()
   
    train_none.append(model.train_curve)
    test_none.append(model.test_curve) 
    tf.reset_default_graph()
train_none, test_none = np.array(train_none), np.array(test_none)

batchnorm = 'standard' 
for run in range(runs):
    print("Starting work on run: {}".format(run))
    seed = seeds[run]
    np.random.seed(seed)
    ## load data 
    data = Data(mnist_data.x_train, mnist_data.y_train, x_test=mnist_data.x_test, y_test=mnist_data.y_test, dataset='mnist', seed=seed)
    n_x, n_y = data.INPUT_DIM, data.NUM_CLASSES
    
    model = bnn(n_x, n_y, n_hidden, 'categorical', initVar=initVar, batchnorm=batchnorm, wSamples=1)
    model.train(data, n_epochs, batchsize, lr, eval_samps=eval_samps, binarize=True) 
 
    train_bn.append(model.train_curve)
    test_bn.append(model.test_curve) 
    tf.reset_default_graph()
train_bn, test_bn = np.array(train_bn), np.array(test_bn)

batchnorm = 'bayes' 
for run in range(runs):
    print("Starting work on run: {}".format(run))
    seed = seeds[run]
    np.random.seed(seed)
    ## load data 
    data = Data(mnist_data.x_train, mnist_data.y_train, x_test=mnist_data.x_test, y_test=mnist_data.y_test, dataset='mnist', seed=seed)
    n_x, n_y = data.INPUT_DIM, data.NUM_CLASSES
    
    model = bnn(n_x, n_y, n_hidden, 'categorical', initVar=initVar, batchnorm=batchnorm, wSamples=1)
    model.train(data, n_epochs, batchsize, lr, eval_samps=eval_samps, binarize=True)

    train_bbn.append(model.train_curve)
    test_bbn.append(model.test_curve) 
    tf.reset_default_graph()
train_bbn, test_bbn = np.array(train_bbn), np.array(test_bbn)


### Plot results ###
plt.figure()
plt.plot(test_none.mean(0), color='r', label='No BN')
plt.plot(test_bn.mean(0), color='m', label='Regular BN')
plt.plot(test_bbn.mean(0), color='b', label='Bayesian BN')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./plots/mnist_convergence', bbox_inches='tight')




