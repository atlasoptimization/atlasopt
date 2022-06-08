"""
The goal of this script is to showcase kernel inference for the task of estimating
the covariance of a function. It exhibits an instationary correlation structure. 
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
import scipy.linalg as spla
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

# np.random.seed(1)                   # Activate line for reproducibility 


# ii) Definition of auxiliary quantities

n=100
n_sample=10
n_simu=1000

t=np.linspace(0,1,n)
sample_index=np.round(np.linspace(0,n-1,n_sample))
t_sample=t[sample_index.astype(int)]

tol=10**(-6)



"""
    2. Create covariance matrices --------------------------------------------
"""


# i) Define auxiliary covariance function

d_sqexp=0.05
def cov_fun_sqexp(t1,t2):
    return (1/n**2)*np.exp(-(lina.norm(t1-t2)/d_sqexp)**2)

K_sqexp=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        K_sqexp[k,l]=cov_fun_sqexp(t[k],t[l])


# ii) Introduce constrained behavior

Nabla=np.delete(np.eye(n)-np.roll(np.eye(n),1,1),n-1,0)
Delta=np.delete(Nabla.T@Nabla,[0,n-1],0)

L=np.zeros([3,n])
L[0,0]=1; L[2,n-1]=1; L[1,np.round(n/2-1).astype(int)]=1

A_constraints=np.vstack((Delta,L))
K_sqexp_mod=np.delete(K_sqexp,[0,n-1],0)
K_sqexp_mod=np.delete(K_sqexp_mod,[0,n-1],1)
K_constrained=spla.block_diag(K_sqexp_mod,np.zeros([3,3]))


# iii) Solve A_c K_x A_c.T=K_c

K_x=lina.pinv(A_constraints)@K_constrained@lina.pinv(A_constraints).T



"""
    3. Simulation of autocorrelated data -------------------------------------
"""


# i) Draw from a distribution with covariance matrix K_x

x_simu=np.zeros([n,n_simu])
for k in range(n_simu):
    x_simu[:,k]=np.random.multivariate_normal(np.zeros([n]),K_x)

x_measured=x_simu[sample_index.astype(int),:]

S_emp=(1/n_simu)*(x_simu@x_simu.T)
S_emp_measured=(1/n_simu)*(x_measured@x_measured.T)




"""
    4. Kernel inference ------------------------------------------------------
"""


# i) Preparation

r=1
n_exp=10


d_prior=0.2
def cov_fun_prior(t1,t2):
    return np.exp(-(lina.norm(t1-t2)/d_prior)**2)

K_prior=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        K_prior[k,l]=cov_fun_prior(t[k],t[l])

[U_p,Lambda_p,V_p]=lina.svd(K_prior,hermitian=True)
U_p_cut=U_p[:,:n_exp]
Psi=U_p_cut[sample_index.astype(int),:]
Lambda_p_cut=np.diag(Lambda_p[:n_exp])


# ii) Execute inference

import OE_KI as KI
beta, mu, gamma, C_gamma, KI_logfile = KI.Kernel_inference_homogeneous(x_measured,Lambda_p_cut,Psi,r)



"""
    5. Optimal estimation  ---------------------------------------------------
"""


# i) Auxiliary quantities

n_datapoints= 4
datapoint_index=np.sort(np.random.choice(range(n),size=n_datapoints))
t_datapoints=t[datapoint_index.astype(int)]
x_datapoints=x_simu[datapoint_index.astype(int),:]


# ii) Interpolate using squared exponential

d_sqexp_interpolate=0.2
def cov_fun_sqexp_interpolate(t1,t2):
    return np.exp(-(lina.norm(t1-t2)/d_sqexp_interpolate)**2)

K_sqexp_interpolate_sample=np.zeros([n_datapoints,n_datapoints])
K_sqexp_interpolate_subset=np.zeros([n,n_datapoints])
for k in range(n_datapoints):
    for l in range(n_datapoints):
        K_sqexp_interpolate_sample[k,l]=cov_fun_sqexp_interpolate(t_datapoints[k],t_datapoints[l])
        
for k in range(n):
    for l in range(n_datapoints):
        K_sqexp_interpolate_subset[k,l]=cov_fun_sqexp_interpolate(t[k],t_datapoints[l])
        
x_est_K_sqexp=K_sqexp_interpolate_subset@lina.pinv(K_sqexp_interpolate_sample,rcond=tol,hermitian=True)@x_datapoints


# iii) Interpolate using inferred kernel

K_gamma=U_p_cut@gamma@U_p_cut.T
K_gamma_sample=K_gamma[np.ix_(datapoint_index.astype(int),datapoint_index.astype(int))]
K_gamma_subset=K_gamma[:,datapoint_index.astype(int)]

x_est_K_gamma=K_gamma_subset@lina.pinv(K_gamma_sample,rcond=tol,hermitian=True)@x_datapoints


# iv) Interpolate using true kernel

K_true_sample=K_x[np.ix_(datapoint_index.astype(int),datapoint_index.astype(int))]
K_true_subset=K_x[:,datapoint_index.astype(int)]

x_est_K_true=K_true_subset@lina.pinv(K_true_sample,rcond=tol,hermitian=True)@x_datapoints




"""
    6. Plots and illustrations -----------------------------------------------
"""


# i) Auxiliary definitions

zero_line=np.zeros([n,1])


# ii) Invoke figure 1

n_plot=15
w,h=plt.figaspect(0.4)
fig1 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs1 = fig1.add_gridspec(1, 3)


# Location 1,1 Example realizations
f1_ax1 = fig1.add_subplot(gs1[0,0])
f1_ax1.plot(t,x_simu[:,1:n_plot],linestyle='solid',color='0',label='Estimate sqexp')

y_min,y_max=plt.ylim()
plt.vlines(t_sample,y_min,y_max,color='0.75',linestyle='dashed')
plt.ylabel('Function value x(t)')
plt.xlabel('Time t')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax1.set_title('Example realizations')


# # Location 1,2 Plot of the empirical covariance matrix
# f1_ax2 = fig1.add_subplot(gs1[0,1])
# f1_ax2.imshow((1/n_sample)*x_measured@x_measured.T)
# plt.ylabel('x direction')
# plt.xlabel('x direction')
# plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
# plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
# f1_ax2.set_title('Empirical covariance')


# Location 1,2 Plot of the inferred covariance function
f1_ax2 = fig1.add_subplot(gs1[0,1])
f1_ax2.imshow(K_gamma)
plt.ylabel('x direction')
plt.xlabel('x direction')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
f1_ax2.set_title('Estimated covariance')



# Location 1.3 Estimations using inferred covariance

n_illu=1

f1_ax3 = fig1.add_subplot(gs1[0,2])
f1_ax3.scatter(t_datapoints,x_datapoints[:,0],facecolors='none',edgecolors='0',label='Data points')
for k in range(n_illu-1):
    f1_ax3.scatter(t_datapoints,x_datapoints[:,k+1],facecolors='none',edgecolors='0')
    
gamma_est = f1_ax3.plot(t,x_est_K_gamma[:,:n_illu],linestyle='solid',color='0',label='Estimate inferred cov')
plt.setp(gamma_est[1:], label="_")
true_est = f1_ax3.plot(t,x_simu[:,:n_illu],linestyle='dotted',color='0.65',label='True process')
plt.setp(true_est[1:], label="_")
f1_ax3.plot(t,zero_line,linestyle='dotted',color='0.5')
plt.tick_params(axis='y', which='both', left=False,right=False,labelleft=False)
plt.tick_params(axis='x', which='both', top=False,bottom=False,labelbottom=False)
plt.xlabel('Time t')
f1_ax3.set_title('Estimations and truth')
# f1_ax3.legend(loc='lower right')






















