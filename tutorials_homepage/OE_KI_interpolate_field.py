"""
The goal of this script is to showcase kernel inference for the task of estimating
the covariance of a 2d field. It exhibits an instationary correlation structure. 
This produces figures showcasing  the kernel inference procedure and its uses 
for multivariate estimation.

For this, do the following:
    1. Imports and definitions
    2. Create covariance matrices
    3. Simulation of autocorrelated data
    4. Kernel inference
    5. Optimal estimation
    6. Plots and illustrations
    
The simulations are based on a fixed random seed, to generate data that is 
different for each run, please comment out the entry 'np.random.seed(x)' in 
section 1.

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""



"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import numpy as np
import numpy.linalg as lina
import matplotlib.pyplot as plt
import OE_KI_support_funs as sf
plt.rcParams.update({'font.size': 12})

# np.random.seed(2)                     # Activate line for reproducibility 


# ii) Definition of auxiliary quantities

n_x=25
n_y=25
n_tot=n_x*n_y

n_sample_x=20
n_sample_y=20
n_sample=n_sample_x*n_sample_y

n_simu=100

t_x=np.linspace(0,1,n_x)
t_y=np.linspace(0,1,n_y)

grid_x,grid_y=np.meshgrid(t_x,t_y)

sample_index_x=np.round(np.linspace(0,n_x-1,n_sample_x))
sample_index_y=np.round(np.linspace(0,n_y-1,n_sample_y))
t_x_sample=t_x[sample_index_x.astype(int)]
t_y_sample=t_y[sample_index_y.astype(int)]

tol=10**(-6)



"""
    2. Create covariance matrices --------------------------------------------
"""


# i) Define covariance functions

d_x=0.1
def cov_fun_x(t1,t2):
    return np.exp(-(lina.norm(t1-t2)/d_x)**2)

d_y=0.2
def cov_fun_y(t1,t2):
    return np.exp(-(lina.norm(t1-t2)/d_y)**2)


# ii) Derive component covariance matrices

K_x=np.zeros([n_x,n_x])
for k in range(n_x):
    for l in range(n_x):
        K_x[k,l]=cov_fun_x(t_x[k],t_x[l])
        
        
K_y=np.zeros([n_y,n_y])
for k in range(n_y):
    for l in range(n_y):
        K_y[k,l]=cov_fun_y(t_y[k],t_y[l])


[U_x,S_x,V_x]=lina.svd(K_x)
[U_y,S_y,V_y]=lina.svd(K_y)



"""
    3. Simulation of autocorrelated data -------------------------------------
"""


# i) Draw from a distribution with covariance matrix K_x

explained_var=0.95
Random_field_collection=np.zeros([n_y,n_x,n_simu])
x_measured=np.zeros([n_sample,n_simu])
for k in range(n_simu):
    
    Random_field_temp=sf.Simulation_random_field(cov_fun_x, cov_fun_y, grid_x, grid_y, explained_var)
    Random_field_collection[:,:,k]=Random_field_temp
    x_measured_temp=Random_field_temp[np.ix_(sample_index_y.astype(int),sample_index_x.astype(int))]
    x_measured[:,k]=np.ravel(x_measured_temp)
    
S_emp_measured=(1/n_sample)*x_measured@x_measured.T

"""
    4. Kernel inference ------------------------------------------------------
"""


# i) Preparation

r=2
n_exp=30


# iii) Prior and basis

lambda_mat=np.outer(S_y, S_x)
index_mat_ordered=np.unravel_index(np.argsort(-lambda_mat.ravel()), [n_y,n_x])
lambda_ordered=lambda_mat[index_mat_ordered]

lambda_tot=np.sum(lambda_mat)
lambda_cumsum=np.cumsum(lambda_ordered)
stop_index=n_exp

Lambda_p_cut=np.diag(lambda_ordered[:n_exp])

U_p_cut=np.zeros([n_tot,n_exp])
Psi=np.zeros([n_sample,n_exp])
for k in range(n_exp):
    basis_fun=np.outer(U_y[:,index_mat_ordered[0][k]],U_x[:,index_mat_ordered[1][k]])
    U_p_cut[:,k]=np.ravel(basis_fun)
    Psi[:,k]=np.ravel(basis_fun[np.ix_(sample_index_y.astype(int),sample_index_x.astype(int))])


# ii) Execute inference

import OE_KI as KI
beta, mu, gamma, C_gamma, KI_logfile = KI.Kernel_inference_homogeneous(x_measured,Lambda_p_cut,Psi,r)



"""
    5. Optimal estimation  ---------------------------------------------------
"""


# i) Auxiliary quantities

n_datapoints= 15
datapoint_index_x=np.random.choice(range(n_x),size=n_datapoints)
datapoint_index_y=np.random.choice(range(n_y),size=n_datapoints)
x_datapoints=grid_x[datapoint_index_y.astype(int),datapoint_index_x.astype(int)]
y_datapoints=grid_y[datapoint_index_y.astype(int),datapoint_index_x.astype(int)]
rf_datapoints=Random_field_temp[datapoint_index_y.astype(int),datapoint_index_x.astype(int)]


# ii) Interpolate using squared exponential

d_sqexp_interpolate=0.3
def cov_fun_exp_interpolate(t1,t2):
    return np.exp(-(lina.norm(t1-t2)/d_sqexp_interpolate)**1)

K_sqexp_interpolate_sample=np.zeros([n_datapoints,n_datapoints])
K_sqexp_interpolate_subset=np.zeros([n_tot,n_datapoints])
for k in range(n_datapoints):
    for l in range(n_datapoints):
        t1=np.array([grid_x[datapoint_index_y[k],datapoint_index_x[k]],grid_y[datapoint_index_y[k],datapoint_index_x[k]]])
        t2=np.array([grid_x[datapoint_index_y[l],datapoint_index_x[l]],grid_y[datapoint_index_y[l],datapoint_index_x[l]]])
        K_sqexp_interpolate_sample[k,l]=cov_fun_exp_interpolate(t1,t2)
        
for k in range(n_tot):
    for l in range(n_datapoints):
        t1=np.array([np.matrix.flatten(grid_x)[k],np.matrix.flatten(grid_y)[k]])
        t2=np.array([grid_x[datapoint_index_y[l],datapoint_index_x[l]],grid_y[datapoint_index_y[l],datapoint_index_x[l]]])
        K_sqexp_interpolate_subset[k,l]=cov_fun_exp_interpolate(t1,t2)
        
rf_est_K_exp=np.reshape(K_sqexp_interpolate_subset@lina.pinv(K_sqexp_interpolate_sample,rcond=tol,hermitian=True)@rf_datapoints,[n_y,n_x])


# iii) Interpolate using inferred kernel

datapoint_index=np.ravel_multi_index((datapoint_index_y,datapoint_index_x),[n_y,n_x])

K_gamma=U_p_cut@gamma@U_p_cut.T
K_gamma_sample=K_gamma[np.ix_(datapoint_index.astype(int),datapoint_index.astype(int))]
K_gamma_subset=K_gamma[:,datapoint_index.astype(int)]

rf_est_K_gamma=np.reshape(K_gamma_subset@lina.pinv(K_gamma_sample,rcond=tol,hermitian=True)@rf_datapoints,[n_y,n_x])


# iv) Interpolate using true kernel

K_true=np.kron(K_y,K_x)
K_true_sample=K_true[np.ix_(datapoint_index.astype(int),datapoint_index.astype(int))]
K_true_subset=K_true[:,datapoint_index.astype(int)]

rf_est_K_true=np.reshape(K_true_subset@lina.pinv(K_true_sample,rcond=tol,hermitian=True)@rf_datapoints,[n_y,n_x])




"""
    6. Plots and illustrations -----------------------------------------------
"""



# i) Invoke figure 1

n_plot=15
w,h=plt.figaspect(0.4)
fig1 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs1 = fig1.add_gridspec(1, 3)


# Location 1,1 Example realizations
f1_ax1 = fig1.add_subplot(gs1[0,0])

plt.imshow(Random_field_temp,extent=[0,1,0,1])
plt.ylabel('Location y')
plt.xlabel('Location x')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax1.set_title('Example realization')


# Location 1,2 Estimated covariance function
f1_ax2 = fig1.add_subplot(gs1[0,1])
f1_ax2.imshow(C_gamma)
plt.ylabel('Locations x,y')
plt.xlabel('Locations x,y')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax2.set_title('Estimated covariance')


# Location 1.3 Estimations using inferred covariance
f1_ax3 = fig1.add_subplot(gs1[0,2])
f1_ax3.scatter(x_datapoints,y_datapoints,facecolors='1',edgecolors='1',label='Data points')
    
gamma_est = f1_ax3.imshow(np.flipud(rf_est_K_gamma),extent=[0,1,0,1],label='Estimate inferred cov')
plt.setp(gamma_est, label="_")
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
plt.xlabel('Location x')
f1_ax3.set_title('Estimations and truth')


# ii) Invoke figure 2 (only contains ground truth))

n_plot=15
w,h=plt.figaspect(0.4)
fig2 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs2 = fig2.add_gridspec(1, 3)

f2_ax3 = fig2.add_subplot(gs1[0,2])
f2_ax3.scatter(x_datapoints,y_datapoints,facecolors='1',edgecolors='1',label='Data points')
    
gamma_est = f2_ax3.imshow(np.flipud(Random_field_temp),extent=[0,1,0,1],label='Ground truth')
plt.setp(gamma_est, label="_")
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
plt.xlabel('Location x')
f1_ax3.set_title('Ground truth')

















