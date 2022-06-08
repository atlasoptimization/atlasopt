"""
The goal of this script is to demonstrate the non-uniqueness of solutions to 
the interpolation problem in a simple 1D setting. We do this by randomly 
generating some data points and then performing conditional simulation to show
different process realizations consistent with the data. The optimal estimator
is illustrated as well.
For this, do the following:
    1. Imports and definitions
    2. Randomly generate data
    3. Conditional simulation
    4. Create the optimal estimate
    5. Plots and illustratons

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""



"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt


# ii) Definitions - dimensions

n_sample=4             # nr of measurements
n_disc=100              # nr of points


# iii) Definitions - auxiliary quantities

x=np.linspace(0,1,n_disc)



"""
    2. Randomly generate data ------------------------------------------------
"""


# i) Measurement locations

np.random.seed(0)

sample_index=np.random.choice(np.linspace(0,n_disc-1,n_disc),[n_sample,1], replace=False).astype(int)
x_sample=x[sample_index]


# ii) Create correlation structures

d_x=0.3
d_y=0.1


cov_1=lambda x1,x2: np.exp(-((x1-x2)/d_x)**2)       # Squared exponential covariance
cov_2=lambda x1,x2: 0.2*np.exp(-((x1-x2)/d_y)**2)
# cov_2=lambda x1,x2: np.exp(-(np.abs(x1-x2)/d_y))  # Ornstein Uhlenbeck covariance
# cov_3=lambda x1,x2: np.min(np.array([[x1],[x2]],dtype=object))-x1*x2           # Browinian bridge covariance
cov_3=lambda x1,x2: np.min(np.array([[x1],[x2]],dtype=object))+0.5           # Browinian bridge covariance

K_0=np.eye(n_disc)
K_1=np.zeros([n_disc,n_disc])
K_2=np.zeros([n_disc,n_disc])
K_3=np.zeros([n_disc,n_disc])

for k in range(n_disc):
    for l in range(n_disc):
       K_1[k,l]=cov_1(x[k],x[l])     
       K_2[k,l]=cov_2(x[k],x[l])
       K_3[k,l]=cov_3(x[k],x[l])


# iii) Simulate the data

mu_full=np.zeros([n_disc])
mu_sample=np.zeros([n_sample,1])

proc=np.random.multivariate_normal(mu_full,K_1)
proc_sample=proc[sample_index]



"""
    3. Conditional simulation ------------------------------------------------
"""


# i) Conditional distribution

# Create Sigma 11, Sigma 12, Sigma 22
K_ij_1=np.zeros([n_sample,n_sample])
K_ij_2=np.zeros([n_sample,n_sample])
K_ij_3=np.zeros([n_sample,n_sample])

K_t_1=np.zeros([n_disc,n_sample])
K_t_2=np.zeros([n_disc,n_sample])
K_t_3=np.zeros([n_disc,n_sample])

for k in range(n_sample):
    for l in range(n_sample):
        K_ij_1[k,l]=cov_1(x[sample_index[k]], x[sample_index[l]])
        K_ij_2[k,l]=cov_2(x[sample_index[k]], x[sample_index[l]])
        K_ij_3[k,l]=cov_3(x[sample_index[k]], x[sample_index[l]])

for k in range(n_disc):
    for l in range(n_sample):
        K_t_1[k,l]=cov_1(x[k], x[sample_index[l]])
        K_t_2[k,l]=cov_2(x[k], x[sample_index[l]])
        K_t_3[k,l]=cov_3(x[k], x[sample_index[l]])


# ii) Prepare conditional simulation

Sigma_11_1=K_1; Sigma_11_2=K_2; Sigma_11_3=K_3
Sigma_22_1=K_ij_1; Sigma_22_2=K_ij_2; Sigma_22_3=K_ij_3
Sigma_12_1=K_t_1; Sigma_12_2=K_t_2; Sigma_12_3=K_t_3

mu_bar_1=np.reshape(mu_full,[n_disc,1])+Sigma_12_1@np.linalg.pinv(Sigma_22_1)@(proc_sample-mu_sample)
Sigma_bar_1=Sigma_11_1-Sigma_12_1@np.linalg.pinv(Sigma_22_1)@Sigma_12_1.T

mu_bar_2=np.reshape(mu_full,[n_disc,1])+Sigma_12_2@np.linalg.pinv(Sigma_22_2)@(proc_sample-mu_sample)
Sigma_bar_2=Sigma_11_2-Sigma_12_2@np.linalg.pinv(Sigma_22_2)@Sigma_12_2.T

mu_bar_3=np.reshape(mu_full,[n_disc,1])+Sigma_12_3@np.linalg.pinv(Sigma_22_3)@(proc_sample-mu_sample)
Sigma_bar_3=Sigma_11_3-Sigma_12_3@np.linalg.pinv(Sigma_22_3)@Sigma_12_3.T


# iii) Simulate the data

n_simu=5
simu_1=np.zeros([n_disc,n_simu])
simu_2=np.zeros([n_disc,n_simu])
simu_3=np.zeros([n_disc,n_simu])

for k in range(n_simu):
    simu_1[:,k]=np.random.multivariate_normal(mu_bar_1.flatten(), Sigma_bar_1)
    simu_2[:,k]=np.random.multivariate_normal(mu_bar_2.flatten(), Sigma_bar_2)
    simu_3[:,k]=np.random.multivariate_normal(mu_bar_3.flatten(), Sigma_bar_3)



"""
    4. Create the optimal estimate --------------------------------------------
"""


# i) Optimal estimate

x_hat=K_t_1@np.linalg.pinv(K_ij_1)@proc_sample



"""
    5. Plots and illustratons ------------------------------------------------
"""


# i) Plot conditional simulation cov 1

padding=0.2

plt.figure(1,dpi=300)
plt.plot(x,simu_1,color='k')
plt.scatter(x_sample, proc_sample, label='data',facecolors='w',s=80, lw=3,  color='k', zorder=3)
y_min, y_max=plt.ylim()
plt.ylim(y_min-padding,y_max+padding)
plt.title('Conditional simulation')
plt.xlabel('z axis')
plt.ylabel('function value')
plt.legend()


# ii) Plot conditional simulation cov 2

plt.figure(2,dpi=300)
plt.plot(x,simu_2,color='k')
plt.scatter(x_sample, proc_sample, label='data',facecolors='w',s=80, lw=3,  color='k', zorder=3)
plt.ylim(y_min-padding,y_max+padding)
plt.title('Conditional simulation')
plt.xlabel('z axis')
plt.ylabel('function value')
plt.legend()


# iii) Plot conditional simulation cov 3

plt.figure(3,dpi=300)
plt.plot(x,simu_3,color='k')
plt.scatter(x_sample, proc_sample, label='data',facecolors='w',s=80, lw=3,  color='k', zorder=3)
plt.ylim(y_min-padding,y_max+padding)
plt.title('Conditional simulation')
plt.xlabel('z axis')
plt.ylabel('function value')
plt.legend()


# iv) Plot optimal estimate

plt.figure(4,dpi=300)
plt.plot(x,x_hat,color='k')
plt.scatter(x_sample, proc_sample, label='data',facecolors='w',s=80, lw=3,  color='k', zorder=3)
plt.ylim(y_min-padding,y_max+padding)
plt.title('Optimal estimate')
plt.xlabel('z axis')
plt.ylabel('function value')
plt.legend()


# v) Plot covariance matrix

plt.figure(5,dpi=300)
plt.imshow(K_1)
plt.title('Covariance matrix')
plt.xlabel('z axis')
plt.ylabel('z axis')










