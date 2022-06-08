"""
The goal of this script is to demonstrate optimal estimation in some simple 2D
settings. The tasks to be solved include regression, interpolation, signal 
separation and an illustration of uncertainty. 
For this, do the following:
    1. Imports and definitions
    2. Randomly generate data
    3. Formulate and solve the estimation problem
    4. Plots and illustratons

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""



"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import numpy as np
from OE_simulation_support_funs import Simulation_random_field
import matplotlib.pyplot as plt


# ii) Definitions - dimensions

n_sample=10             # nr of measurements
n_dim_theta=3           # nr of unknowns in the model A*theta=y
n_disc=50              # nr of points in x, y direction
n_total=n_disc**2

# iii) Definitions - auxiliary quantities

x=np.linspace(0,1,n_disc)
y=x

grid_x, grid_y=np.meshgrid(x,y)
ss=np.vstack((grid_x.flatten(), grid_y.flatten()))

d_x=0.3
d_y=0.3

cov_x=lambda x1,x2: np.exp(-((x1-x2)/d_x)**2)
cov_y=lambda y1,y2: np.exp(-((y1-y2)/d_y)**2)



"""
    2. Randomly generate data ------------------------------------------------
"""


# i) Measurement locations

np.random.seed(1)

index_sequence_1d=np.linspace(0,n_disc,n_disc).astype(int)
ii1, ii2=np.meshgrid(index_sequence_1d,index_sequence_1d)
index_sequence_2d=np.vstack((ii1.flatten(), ii2.flatten()))

rand_int=np.random.choice(np.linspace(0,n_total-1, n_total).astype(int), size=[n_sample,1], replace=False)
index_tuples=np.unravel_index(rand_int, [n_disc,n_disc])


# ii) Simulate the data

RF, K_x, K_y=Simulation_random_field(cov_x, cov_y, grid_x, grid_y, 0.95)
RF_image=np.rot90(RF)
ss_sample=np.zeros([2,n_sample])
for k in range(n_sample):
    ss_sample[0,k]=x[index_tuples[0][k]]
    ss_sample[1,k]=y[index_tuples[1][k]]

l_sample=RF[index_tuples].T


# iii) Correlation data

K_ij=np.zeros([n_sample,n_sample])
K_t=np.zeros([n_total,n_sample])

for k in range(n_sample):
    for l in range(n_sample):
        K_ij[k,l]=cov_x(x[index_tuples[0][k]],x[index_tuples[0][l]])*cov_y(y[index_tuples[1][k]],y[index_tuples[1][l]])

for k in range(n_total):
    for l in range(n_sample):
        K_t[k,l]=cov_x(ss[0,k],x[index_tuples[0][l]])*cov_y(ss[1,k],y[index_tuples[1][l]])



"""
    3. Formulate and solve the estimation problem -----------------------------
"""


# i) Interpolation problem

RF_hat=np.reshape(K_t@np.linalg.pinv(K_ij)@l_sample.T,[n_disc,n_disc],order='F')
RF_interpolation_image=np.rot90(RF_hat)


# ii) Regression problem

# Regressor functions

n_reg=7
g00=lambda x,y: 1
g10=lambda x,y: x
g01=lambda x,y: y
g11=lambda x,y: x*y
g20=lambda x,y: x**2
g21=lambda x,y: x**2*y
g02=lambda x,y: y**2
g12=lambda x,y: y**2*x
g22= lambda x,y: x**2*y**2

reg_list=[g00,g10,g01,g11,g20, g21,g02, g12,g22]


# Design matrix
A=np.zeros([n_sample,n_reg])
for k in range(n_sample):
    for l in range(n_reg):
        A[k,l]=reg_list[l](ss_sample[0,k],ss_sample[1,k])
        
A_full=np.zeros([n_total,n_reg])
for k in range(n_total):
    for l in range(n_reg):
        A_full[k,l]=reg_list[l](ss[0,k],ss[1,k])
        
# Parameters and estimator
theta_hat=np.linalg.pinv(A)@l_sample.T
Rf_hat_regression=np.reshape(A_full@theta_hat,[n_disc,n_disc],order='F')
RF_regression_image=np.rot90(Rf_hat_regression)

components=np.zeros([n_disc,n_disc,n_reg])
for k in range(n_reg):
    components[:,:,k]=np.rot90(np.reshape(A_full[:,k]*theta_hat[k],[n_disc,n_disc],order='F'))




"""
    4. Plots and illustratons ------------------------------------------------
"""


# i) Plot interpolation of measurements

plt.figure(1,dpi=300)
plt.scatter(ss_sample[0,:], ss_sample[1,:],  s=90, facecolors='w', edgecolors='k',linewidths=2)
plt.imshow(RF_interpolation_image,extent=[0,1,0,1])
plt.title('Interpolation')
plt.xlabel('x axis')
plt.ylabel('y axis')


# ii) Plot regression of measurements

plt.figure(2,dpi=300)
plt.scatter(ss_sample[0,:], ss_sample[1,:],  s=90,facecolors='w', edgecolors='k',linewidths=2)
plt.imshow(RF_regression_image,extent=[0,1,0,1])
plt.title('Regression')
plt.xlabel('x axis')
plt.ylabel('y axis')

# iii) Plot ground truth

plt.figure(3,dpi=300)
plt.scatter(ss_sample[0,:], ss_sample[1,:], s=90, facecolors='w', edgecolors='k',linewidths=2)
plt.imshow(RF_image,extent=[0,1,0,1])
plt.title('True underlying field')
plt.xlabel('x axis')
plt.ylabel('y axis')


# iv) Plot correlation 1d

plt.figure(4,dpi=300)
plt.imshow(K_x, extent=[0,1,0,1])
plt.title('Covariance matrix 1D')
plt.xlabel('x axis')
plt.ylabel('x axis')


# v) Plot correlation 2d

plt.figure(5,dpi=300)
plt.imshow(np.kron(K_x,K_y))
plt.title('Covariance marix 2D')
plt.xlabel('(x,y) axis')
plt.ylabel('(x,y) axis')
plt.xticks([])
plt.yticks([])


# vi) Plot signal separation of measurements

plt.figure(6,dpi=300)
plt.imshow(components[:,:,1],extent=[0,1,0,1])
plt.title('Component 1')
plt.xlabel('x axis')
plt.ylabel('y axis')

plt.figure(7,dpi=300)
plt.imshow(components[:,:,2],extent=[0,1,0,1])
plt.title('Component 2')
plt.xlabel('x axis')
plt.ylabel('y axis')

plt.figure(8,dpi=300)
plt.imshow(components[:,:,3],extent=[0,1,0,1])
plt.title('Component 3')
plt.xlabel('x axis')
plt.ylabel('y axis')

plt.figure(9,dpi=300)
plt.imshow(components[:,:,5],extent=[0,1,0,1])
plt.title('Component 4')
plt.xlabel('x axis')
plt.ylabel('y axis')

plt.figure(10,dpi=300)
plt.imshow(components[:,:,6],extent=[0,1,0,1])
plt.title('Component 5')
plt.xlabel('x axis')
plt.ylabel('y axis')















