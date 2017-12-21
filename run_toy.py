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
from models.mlp import NeuralNet
import sys, pdb
### Script to run a BNN experiment over the moons data

# argv[1] - train teacher (yes, no) 

def generate_teacher_data(numPoints, seed=123):
    np.random.seed(seed)
    mu, covMat= np.array([1.,1.]), [[.5,-.25],[-.25,.5]]
    x1 = np.random.multivariate_normal(mu, covMat, size=(numPoints,))
    x2 = np.random.multivariate_normal(-mu, covMat, size=(numPoints,))
    y = np.squeeze(np.vstack((np.zeros((numPoints,1)), np.ones((numPoints,1)))))
    x, y = np.vstack((x1,x2)), np.eye(2)[y.astype('int')]
    return Data(x, y, x_test=x, y_test=y, dataset='toy_data')
    
def generate_student_data(numPoints, seed=123):
    x = np.random.uniform(-10, 10, size=(numPoints,2))
    y = np.zeros((numPoints,2))
    return Data(x, y, x_test=x, y_test=y, dataset='compress_data')
    	
train_teacher = (sys.argv[1] == 'yes')
# Ranges
range_x = np.arange(-10.,10.,.1)
range_y = np.arange(-10.,10.,.1)
X,Y = np.mgrid[-10.:10.:.1, -10.:10.:.1]
xy = np.vstack((X.flatten(), Y.flatten())).T


## Specify model parameters
lr = (1e-2,)
n_hid_teacher = [25]
n_hid_student = [10]
n_epochs, batchsize = 10, 8 
initVar, eval_samps = -9.0, None

batchnorm = 'None' 
## load data 
teacher_data = generate_teacher_data(10) 
student_data = generate_student_data(5000)
n_x, n_y = teacher_data.INPUT_DIM, teacher_data.NUM_CLASSES

teacher = bnn(n_x, n_y, n_hid_teacher, 'categorical', initVar=initVar, batchnorm=batchnorm, wSamples=3)
if train_teacher:
    teacher.train(teacher_data, n_epochs, batchsize, lr, eval_samps=eval_samps, binarize=False)
else:
    teacher.data_init(teacher_data, eval_samps)


n_x, n_y = student_data.INPUT_DIM, student_data.NUM_CLASSES
student = Student(teacher, n_x, n_y, n_hid_student, y_dist='categorical')
training_params = {'n_epochs':50, 'batchsize':256, 'binarize':False, 'lr':(5e-4,)}
student.train(student_data, training_params) 


teacher_ckpt, student_ckpt = teacher.ckpt_dir, student.ckpt_dir
tf.reset_default_graph()
teacher = bnn(n_x, n_y, n_hid_teacher, 'categorical', initVar=initVar, batchnorm=batchnorm, wSamples=1)
teacher.ckpt_dir = teacher_ckpt
teacher_predictions = teacher.predict_new(xy.astype('float32'))

### SKLEARN MLP STUFF
#mlp = NeuralNet(2, [25], 2, 'categorical')
#training_params = {'n_epochs':50, 'batchsize':8, 'binarize':False, 'lr':(1e-2,)}
#mlp.fit(teacher_data, training_params)

print('Plotting student, teacher and MLP')
x,y = teacher_data.data['x_test'], teacher_data.data['y_test']
x0, x1 = x[np.where(y[:,0]==1)], x[np.where(y[:,1]==1)]
cmap = plt.cm.BrBG

#mlp_predictions = mlp.predict_new(xy.astype('float32'))
#
#zmlp = np.zeros(X.shape)
#for i, row_val in enumerate(range_x):
#    for j, col_val in enumerate(range_y):
#        idx = np.intersect1d(np.where(np.isclose(xy[:,0],row_val))[0],np.where(np.isclose(xy[:,1],col_val))[0])
#        zmlp[i,j] = mlp_predictions[idx[0],0] * 100
#
#plt.figure()
#plt.contourf(X,Y,zmlp,cmap=cmap)
#plt.scatter(x0[:,0], x0[:,1], color='black', s=6)
#plt.scatter(x1[:,0], x1[:,1], color='gray', s=6)
#plt.savefig('./plots/mlp_plot', bbox_inches='tight')
#plt.close()

student = Student(teacher, n_x, n_y, n_hid_student, y_dist='categorical')
student.ckpt_dir = student_ckpt
student_predictions = student.predict_new(xy.astype('float32'))

zs, zt = np.zeros(X.shape), np.zeros(X.shape)
for i, row_val in enumerate(range_x):
    for j, col_val in enumerate(range_y):
        idx = np.intersect1d(np.where(np.isclose(xy[:,0],row_val))[0],np.where(np.isclose(xy[:,1],col_val))[0])
        zt[i,j] = teacher_predictions[idx[0],0] * 100
        zs[i,j] = student_predictions[idx[0],0] * 100

plt.figure()
plt.contourf(X,Y,zt,cmap=cmap)
plt.scatter(x0[:,0], x0[:,1], color='black', s=6)
plt.scatter(x1[:,0], x1[:,1], color='gray', s=6)
plt.savefig('./plots/teacher_plot', bbox_inches='tight')
plt.close()

plt.figure()
plt.contourf(X,Y,zs,cmap=cmap)
plt.scatter(x0[:,0], x0[:,1], color='black', s=6)
plt.scatter(x1[:,0], x1[:,1], color='gray', s=6)
plt.savefig('./plots/student_plot', bbox_inches='tight')
plt.close()

