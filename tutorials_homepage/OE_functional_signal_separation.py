"""
The goal of this script is to demonstrate a stochastic signal separation procedure. 
It takes simulated data that is a mixture of several stochastic signals and
tries to unmix it by formulation of the maximum likelihood estimator.
For this, do the following:
    1. Imports and definitions
    2. Randomly generate data
    3. Solve the signal separation problem
    4. Plots and illustratons

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

n_disc=200              # nr of points in x, y direction

# iii) Definitions - auxiliary quantities

t=np.linspace(0,1,n_disc)

d_x=0.3
d_y=0.05

cov_1=lambda t,s: np.exp(-((t-s)/d_x)**2)
cov_2=lambda t,s: 0.5*np.exp(-np.abs(((t-s)/d_y))**2)



"""
    2. Randomly generate data ------------------------------------------------
"""


# i) Measurement locations

np.random.seed(2)

index_sequence_1d=np.linspace(0,n_disc,n_disc).astype(int)


# ii) Simulate the data

K_x=np.zeros([n_disc,n_disc])
K_y=np.zeros([n_disc,n_disc])

for k in range(n_disc):
    for l in range(n_disc):
        K_x[k,l]=cov_1(t[k],t[l])
        K_y[k,l]=cov_2(t[k],t[l])

x_simu=np.random.multivariate_normal(np.zeros([n_disc]),K_x+K_y)



"""
    3. Solve the signal separation problem ------------------------------------
"""


# i) Signal separationm

x_hat=K_x@np.linalg.pinv(K_x+K_y)@x_simu
y_hat=K_y@np.linalg.pinv(K_x+K_y)@x_simu




"""
    4. Plots and illustratons ------------------------------------------------
"""


# i) Plot signal separation

w,h=plt.figaspect(0.4)
fig1 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs1 = fig1.add_gridspec(1, 3)


# Location 1,1 Signal
f1_ax1 = fig1.add_subplot(gs1[0,0])

plt.plot(t,x_simu, color='k')
plt.plot(t,np.zeros([n_disc]),color=[0.5,0.5,0.5],linestyle='--')
plt.ylabel('Function value')
plt.xlabel('Time t')
y_min,y_max=plt.ylim()
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax1.set_title('Mixed signal')


# Location 1,2 First estimate
f1_ax2 = fig1.add_subplot(gs1[0,1])

plt.plot(t,x_hat, color='k')
plt.plot(t,np.zeros([n_disc]),color=[0.5,0.5,0.5],linestyle='--')
plt.ylabel('Function value')
plt.xlabel('Time t')
plt.ylim([y_min,y_max])
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax2.set_title('Estimate component 1')


# Location 1,3 Second estimate
f1_ax3 = fig1.add_subplot(gs1[0,2])

plt.plot(t,y_hat, color='k')
plt.plot(t,np.zeros([n_disc]),color=[0.5,0.5,0.5],linestyle='--')
plt.ylabel('Function value')
plt.xlabel('Time t')
plt.ylim([y_min,y_max])
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax3.set_title('Estimate component 2')




# ii) Plot covariancesn

w,h=plt.figaspect(0.4)
fig2 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs1 = fig2.add_gridspec(1, 3)


# Location 1,1 Signal
f2_ax1 = fig2.add_subplot(gs1[0,0])

plt.imshow(K_x+K_y)
plt.ylabel('Time t')
plt.xlabel('Time t')
plt.clim(0,1.3)
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f2_ax1.set_title('Covariance signal')


# Location 1,2 First estimate
f2_ax2 = fig2.add_subplot(gs1[0,1])

plt.imshow(K_x)
plt.ylabel('Time t')
plt.xlabel('Time t')
plt.clim(0,1.3)
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f2_ax2.set_title('Covariance component 1')


# Location 1,3 Second estimate
f2_ax3 = fig2.add_subplot(gs1[0,2])

plt.imshow(K_y)
plt.ylabel('Time t')
plt.xlabel('Time t')
plt.clim(0,1.3)
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f2_ax3.set_title('Covariance component 2')




