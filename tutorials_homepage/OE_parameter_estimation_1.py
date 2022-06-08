"""
The goal of this script is to illustrate parameter estimation and solve a small
quadratic program illustrating fitting a line
    1. Imports and definitions
    2. Simulate data
    3. Solve the optimal estimation problem
    4. Illustrate

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""



"""
    1. Imports and definitions ------------------------------------------------
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(1)                      # Activate line for reproducibility 


# ii) Definitions

n_z=100
n_sample=10
n_funs=3

z=np.linspace(0,1,n_z)



"""
    2. Simulate data ---------------------------------------------------------
"""


# i) Set up functions

x_true=np.array([1,1,0.3])

g1=lambda z: 1
g2=lambda z: z
g3=lambda z: np.sin(4*np.pi*z)


# ii) Generate full dataset

g1_vec=np.zeros([n_z,1])
g2_vec=np.zeros([n_z,1])
g3_vec=np.zeros([n_z,1])

for k in range(n_z):
    g1_vec[k]=g1(z[k])
    g2_vec[k]=g2(z[k])
    g3_vec[k]=g3(z[k])

noise=np.random.normal(0,0.2,[n_z,1])

data_vec=x_true[0]*g1_vec+x_true[1]*g2_vec+x_true[2]*g3_vec+noise


# ii) Sample data points

rand_ind=np.round(np.random.uniform(0,n_z-1,[n_sample,1])).astype(int)
z_data=np.squeeze(z[rand_ind])
l_data=np.squeeze(data_vec[rand_ind])



"""
    3. Solve the optimal estimation problem ----------------------------------
"""


# i) Create matrices and vectors

G=np.zeros([n_sample, n_funs])
G_full=np.zeros([n_z, n_funs])

for k in range(n_sample):
    G[k,0]=g1(z_data[k])
    G[k,1]=g2(z_data[k])
    G[k,2]=g3(z_data[k])
    
for k in range(n_z):
    G_full[k,0]=g1(z[k])
    G_full[k,1]=g2(z[k])
    G_full[k,2]=g3(z[k])


# ii) Solve for optimal parameters

x_opt=np.linalg.pinv(G)@l_data
l_hat=G_full@x_opt

# iii) Assemble solution



"""
    4. Illustrate ------------------------------------------------------------
"""  
        
        
# i) Plot Realizations

fig, ax = plt.subplots(1, 3, figsize=(10, 3),dpi=500)
ax[0].scatter(z_data,l_data,color='k')
ax[0].set(xlabel='Time t', ylabel='Temperature')
ax[0].set_title('Data')
ax[0].axes.get_xaxis().set_ticks([])
ax[0].axes.get_yaxis().set_ticks([])


ax[1].plot(z,G_full,color='k')
ax[1].set_title('Components of the model')
ax[1].axes.get_xaxis().set_ticks([])
ax[1].axes.get_yaxis().set_ticks([])
ax[1].set(xlabel='Time t')


ax[2].plot(z,l_hat,color='k')
ax[2].scatter(z_data,l_data,color='k')
ax[2].set_title('Optimized model')
ax[2].axes.get_xaxis().set_ticks([])
ax[2].axes.get_yaxis().set_ticks([])
ax[2].set(xlabel='Time t')






