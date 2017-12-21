import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
from sklearn.metrics import log_loss
import tensorflow as tf
import numpy as np
from data.Container import Container as Data
from data.mnist import mnist 
from models.bnn import bnn
from models.student import Student
import sys, pdb
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
n_epochs, batchsize = 3, 1024 
initVar, eval_samps = -10.0, None

train_none, train_bn, train_bbn = [],[],[]
test_none, test_bn, test_bbn = [],[],[]

batchnorm = 'Standard' 
for run in range(runs):
    print("Starting work on run: {}".format(run))
    seed = seeds[run]
    np.random.seed(seed)
    ## load data 
    data = Data(mnist_data.x_train, mnist_data.y_train, x_test=mnist_data.x_test, y_test=mnist_data.y_test, dataset='mnist', seed=seed)
    n_x, n_y = data.INPUT_DIM, data.NUM_CLASSES
    
    teacher = bnn(n_x, n_y, n_hidden, 'categorical', initVar=initVar, batchnorm=batchnorm, wSamples=1)
    teacher.train(data, n_epochs, batchsize, lr, eval_samps=eval_samps, binarize=True)

    student = Student(teacher, n_x, n_y, n_hidden, y_dist='categorical')
    training_params = {'n_epochs':150, 'batchsize':batchsize, 'binarize':True, 'lr':lr}
    student.train(data, training_params) 

    tf.reset_default_graph()

