"""
The goal of this script is to demonstrate different random quantities that all
have some type of correlation structure governing their shape. These include
simple 3D vectors, random functions in 1D, 2D, and vector fields in 2 D.For 
illustrative purposes also the underlying correlation structures are plotted. 
For this, do the following:
    1. Imports and definitions
    2. Randomly generate data: low dim
    3. Randomly generate data: high dim
    4. Plots and illustratons

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""


"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import numpy as np
from scipy.linalg import block_diag
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

cov_x_rvf=lambda x1,x2: np.exp(-((x1-x2)/0.4)**2)
cov_y_rvf=lambda y1,y2: np.exp(-((y1-y2)/0.2)**2)



"""
    2. Randomly generate data: low dim ---------------------------------------
"""


# i) 3 dimensional vectors

np.random.seed(1)
K_v=np.array([[1,0.5, 0.8],[0.5,2,0.4],[0.8,0.4,1.5]])
n_simu=200
random_vector=np.zeros([3,n_simu])
for k in range(n_simu):
    random_vector[:,k]=np.random.multivariate_normal(np.zeros([3]), K_v)


# ii) random functions

n_simu=5
K_x=np.zeros([n_disc,n_disc])
for k in range(n_disc):
    for l in range(n_disc):
        K_x[k,l]=cov_x(x[k],x[l])
        
random_function=np.zeros([n_disc,n_simu])
for k in range(n_simu):
    random_function[:,k]=np.random.multivariate_normal(np.zeros([n_disc]), K_x)



"""
    3. Randomly generate data: high dim --------------------------------------
"""


# i) Simulate the random scalar field

RF, K_x, K_y=Simulation_random_field(cov_x, cov_y, grid_x, grid_y, 0.95)
K_xy_rf=np.kron(K_x,K_y)


# ii) Simulate the random vector field

RVF_1, K_x, K_y=Simulation_random_field(cov_x, cov_y, grid_x, grid_y, 0.95)
RVF_2, K_x_rvf, K_y_rvf=Simulation_random_field(cov_x_rvf, cov_y_rvf, grid_x, grid_y, 0.95)


K_xy_rvf=block_diag(np.kron(K_x,K_y),np.kron(K_x_rvf,K_y_rvf))




"""
    4. Plots and illustratons ------------------------------------------------
"""

# i) 3D vectors

f1=plt.figure(1,dpi=300)
ax = f1.add_subplot(projection='3d')
ax.scatter(random_vector[0,:], random_vector[1,:], random_vector[2,:], color='k')

ax.set_xlabel('x direction')
ax.set_ylabel('y direction')
ax.set_zlabel('z direction')
plt.title('Random vector')
plt.show()

f2=plt.figure(2,dpi=300)
plt.imshow(K_v)
plt.xlabel('Point nr')
plt.ylabel('Point nr')
plt.title('Covariance matrix')


# ii) Random functions

f3=plt.figure(3,dpi=300)
plt.plot(random_function,color='k')
plt.xlabel('x direction')
plt.ylabel('function value')
plt.title('Random function')

f4=plt.figure(4,dpi=300)
plt.imshow(K_x)
plt.xlabel('Point nr')
plt.ylabel('Point nr')
plt.title('Covariance matrix')


# iii) random scalar field

f5=plt.figure(5,dpi=300)
plt.imshow(RF)
plt.xlabel('x direction')
plt.ylabel('y direction')
plt.title('Random 2D field')

f6=plt.figure(6,dpi=300)
plt.imshow(K_xy_rf)
plt.xlabel('Point nr')
plt.ylabel('Point nr')
plt.title('Covariance matrix')


# iv) Random vector field

f7=plt.figure(7,dpi=300)
plt.quiver(grid_x, grid_y, RVF_1, RVF_2)
plt.xlabel('x direction')
plt.ylabel('y direction')
plt.title('Random 2D vector field')

f8=plt.figure(8,dpi=300)
plt.imshow(K_xy_rvf)
plt.xlabel('Point nr')
plt.ylabel('Point nr')
plt.title('Covariance matrix')






























