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

# argv[1] - number of runs (max 10)

def generate_teacher_data(numPoints, seed=123):
    np.random.seed(seed)
    mu, covMat= np.array([3,3]), [[.5,-.25],[-.25,.5]]
    x1 = np.random.multivariate_normal(mu, covMat, size=(numPoints,))
    x2 = np.random.multivariate_normal(-mu, covMat, size=(numPoints,))
    y = np.squeeze(np.vstack((np.zeros((numPoints,1)), np.ones((numPoints,1)))))
    x, y = np.vstack((x1,x2)), np.eye(2)[y.astype('int')]
    return Data(x, y, x_test=x, y_test=y, dataset='toy_data')
    
def generate_student_data(numPoints, seed=123):
    x = np.random.uniform(-10, 10, size=(numPoints,2))
    y = np.zeros((numPoints,2))
    return Data(x, y, x_test=x, y_test=y, dataset='compress_data')
    	
# Ranges
range_x = np.arange(-10.,10.,.1)
range_y = np.arange(-10.,10.,.1)
X,Y = np.mgrid[-10.:10.:.1, -10.:10.:.1]
xy = np.vstack((X.flatten(), Y.flatten())).T


## Specify model parameters
lr = (1e-2,)
n_hidden = [10]
n_epochs, batchsize = 35, 8 
initVar, eval_samps = -7.0, None

batchnorm = 'None' 
## load data 
teacher_data = generate_teacher_data(10) 
student_data = generate_student_data(5000)
n_x, n_y = teacher_data.INPUT_DIM, teacher_data.NUM_CLASSES

teacher = bnn(n_x, n_y, n_hidden, 'categorical', initVar=initVar, batchnorm=batchnorm, wSamples=1)
teacher.train(teacher_data, n_epochs, batchsize, lr, eval_samps=eval_samps, binarize=False)

print('Plotting teacher before student')
teacher_predictions = teacher.predict_new(xy.astype('float32'))
zt = np.zeros(X.shape)

for i, row_val in enumerate(range_x):
    for j, col_val in enumerate(range_y):
        idx = np.intersect1d(np.where(np.isclose(xy[:,0],row_val))[0],np.where(np.isclose(xy[:,1],col_val))[0])
        zt[i,j] = teacher_predictions[idx[0],0] * 100

x,y = teacher_data.data['x_test'], teacher_data.data['y_test']

plt.figure()
plt.contourf(X,Y,zt,cmap=plt.cm.coolwarm)
x0, x1 = x[np.where(y[:,1]==0)], x[np.where(y[:,1]==1)]
plt.scatter(x0[:,0], x0[:,1], color='g', s=1)
plt.scatter(x1[:,0], x1[:,1], color='m', s=1)
plt.savefig('./teacher_plot_before', bbox_inches='tight')
plt.close()

n_x, n_y = student_data.INPUT_DIM, student_data.NUM_CLASSES
student = Student(teacher, n_x, n_y, n_hidden, y_dist='categorical')
training_params = {'n_epochs':35, 'batchsize':256, 'binarize':False, 'lr':(3e-3,)}
student.train(student_data, training_params) 


teacher_ckpt, student_ckpt = teacher.ckpt_dir, student.ckpt_dir
tf.reset_default_graph()
teacher = bnn(n_x, n_y, n_hidden, 'categorical', initVar=initVar, batchnorm=batchnorm, wSamples=1)
teacher.ckpt_dir = teacher_ckpt
teacher_predictions = teacher.predict_new(xy.astype('float32'))


print('Plotting student and teacher')
student = Student(teacher, n_x, n_y, n_hidden, y_dist='categorical')
student.ckpt_dir = student_ckpt
student_predictions = student.predict_new(xy.astype('float32'))

zs, zt = np.zeros(X.shape), np.zeros(X.shape)
for i, row_val in enumerate(range_x):
    for j, col_val in enumerate(range_y):
        idx = np.intersect1d(np.where(np.isclose(xy[:,0],row_val))[0],np.where(np.isclose(xy[:,1],col_val))[0])
        zt[i,j] = teacher_predictions[idx[0],0] * 100
        zs[i,j] = student_predictions[idx[0],0] * 100

plt.figure()
plt.contourf(X,Y,zt,cmap=plt.cm.coolwarm)
plt.savefig('./teacher_plot_final', bbox_inches='tight')
plt.close()

plt.figure()
plt.contourf(X,Y,zs,cmap=plt.cm.coolwarm)
plt.savefig('./student_plot', bbox_inches='tight')
plt.close()

