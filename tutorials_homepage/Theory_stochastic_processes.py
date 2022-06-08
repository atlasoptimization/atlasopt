"""
The goal of this script is to illustrate stochastic processes and solve a small
quadratic program illustrating optimal estimation
For this, do the following:
    1. Imports and definitions
    2. Simulate stochastic processes
    3. Solve the optimal estimation problem
    4. Illustrate

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""



"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# np.random.seed(0)                            # Activate line for reproducibility 


# ii) Definitions

n_t=100
n_simu=2
n_observed=4

t=np.linspace(0,1,n_t)
ind_observed=np.round(np.linspace(0,n_t-1,n_observed)).astype(int)
t_observed=t[ind_observed]



"""
    2. Simulate stochastic processes -----------------------------------------
"""


# i) Build covariance functions

cov_fun_wn= lambda x,y: (x==y)*1
cov_fun_wp= lambda x,y: np.min([x,y])
cov_fun_bb= lambda x,y: np.min([x,y])-x*y
cov_fun_se= lambda x,y: np.exp(-(x-y)**2/0.2)
 

# ii) Strange covariance function

def cov_fun_strange(x,y):
    n_exp=10
    # base_fun=lambda x,k: np.mod(k*x,2)/(k+1) # Sawtooth
    base_fun=lambda x,k: ((x<k/n_exp)*1)/(k+1) # boxes
    
    cov_val=0
    for k in range(n_exp):
        cov_val=cov_val+base_fun(x,k)*base_fun(y,k)    
    return cov_val


# iii) Build covariance matrices

mean_vec=np.zeros(n_t)

K_wn=np.zeros([n_t,n_t])
K_wp=np.zeros([n_t,n_t])
K_bb=np.zeros([n_t,n_t])
K_se=np.zeros([n_t,n_t])
K_strange=np.zeros([n_t,n_t])


for k in range(n_t):
    for l in range(n_t):
        K_wn[k,l]=cov_fun_wn(t[k],t[l])
        K_wp[k,l]=cov_fun_wp(t[k],t[l])
        K_bb[k,l]=cov_fun_bb(t[k],t[l])
        K_se[k,l]=cov_fun_se(t[k],t[l])
        K_strange[k,l]=cov_fun_strange(t[k],t[l])


# iv) Simulate

x_wn=np.zeros([n_simu,n_t])
x_wp=np.zeros([n_simu,n_t])
x_bb=np.zeros([n_simu,n_t])
x_se=np.zeros([n_simu,n_t])
x_strange=np.zeros([n_simu,n_t])

for k in range(n_simu):
    x_wn[k,:]=np.random.multivariate_normal(mean_vec, K_wn)
    x_wp[k,:]=np.random.multivariate_normal(mean_vec, K_wp)
    x_bb[k,:]=np.random.multivariate_normal(mean_vec, K_bb)
    x_se[k,:]=np.random.multivariate_normal(mean_vec, K_se)
    x_strange[k,:]=np.random.multivariate_normal(mean_vec, K_strange)




"""
    3. Solve the optimal estimation problem ----------------------------------
"""


# i) Set up the data

x_se_1=x_se[0,:]
x_observed=x_se_1[ind_observed]

C_mat=np.zeros([n_observed,n_observed])
c_vec=np.zeros([n_observed,n_t])

for k in range(n_observed):
    for l in range (n_observed):
        C_mat[k,l]=cov_fun_se(t_observed[k],t_observed[l])

for k in range(n_observed):
    for l in range(n_t):
        c_vec[k,l]=cov_fun_se(t_observed[k],t[l])



# ii) Define QP and solve one optimal estimation

n_dim_opt_var=n_observed                                      # nr of total optimization variables
opt_var=cp.Variable(n_dim_opt_var)

cons=[cp.sum(opt_var)==1]
objective=cp.Minimize(cp.quad_form(opt_var,C_mat)-2*opt_var.T@c_vec[:,0])

prob=cp.Problem(objective,cons)
prob.solve(verbose=True)

lambda_opt=opt_var.value


# iii) Write down general solution

x_hat=c_vec.T@np.linalg.pinv(C_mat)@x_observed




"""
    4. Illustrate ------------------------------------------------------------
"""  
        
        
# i) Plot Realizations

fig, ax = plt.subplots(1, 4, figsize=(24, 5),dpi=500)
ax[0].plot(t,x_wn.T,color='k')
ax[0].set_title('White noise')
ax[0].axis('off')

ax[1].plot(t,x_wp.T,color='k')
ax[1].set_title('Wiener process')
ax[1].axis('off')

ax[2].plot(t,x_bb.T,color='k')
ax[2].set_title('Brownian bridge')
ax[2].axis('off')

ax[3].plot(t,x_se.T,color='k')
ax[3].set_title('Smooth process')
ax[3].axis('off')


plt.figure(2,dpi=500, figsize=(5,5))
plt.plot(t,x_strange.T,color='k')
plt.title('Strange process')
plt.axis('off')



# ii) Plot estimation

fig, ax = plt.subplots(1, 3, figsize=(10, 5),dpi=500)
ax[0].plot(t,x_se_1.T,color='k')
ax[0].set_title('Original process')
ax[0].axis('off')

ax[1].scatter(t_observed,x_observed,color='k')
ax[1].set_title('Observations')
ax[1].axis('off')

ax[2].plot(t,x_hat.T,color='k')
ax[2].set_title('Estimation of process')
ax[2].axis('off')



# iii) Plot covariance matrices

plt.figure(4, figsize=(5,5), dpi=500)
plt.imshow(K_wn)
plt.title('Covariance white noise')
plt.axis('off')

plt.figure(5, figsize=(5,5), dpi=500)
plt.imshow(K_wp)
plt.title('Covariance wiener process')
plt.axis('off')

plt.figure(6, figsize=(5,5), dpi=500)
plt.imshow(K_bb)
plt.title('Covariance brownian bridge')
plt.axis('off')

plt.figure(7, figsize=(5,5), dpi=500)
plt.imshow(K_se)
plt.title('Covariance smooth')
plt.axis('off')

plt.figure(8, figsize=(5,5), dpi=500)
plt.imshow(K_strange)
plt.title('Covariance strange')
plt.axis('off')



