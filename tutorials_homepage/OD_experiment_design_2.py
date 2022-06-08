"""
The goal of this script is to demonstrate optimal experiment design in a simple
setting. Out of p different vectors v_k, k=1, ..., p representing different 
measurements with associated results 

y_k=(v_k^T theta)+w_k
y_k: measured results, theta: unknown parameters, w_k: white noise

chose n vectors such that they are maximally informative. The degree of information
is gauged by quantifying the size of the covariance matrix of the estimates
of theta. Its trace is demanded to be minimal. 
For this, do the following:
    1. Imports and definitions
    2. Randomly generate matrices
    3. Formulate the semidefinite problem
    4. Assemble the solution
    5. Plots and illustratons

More Information can be found e.g. in pp.511 - 532 Handbook of semidefinite 
programming by H. Wolkowicz et. al., Springer Science & business Media (2003).
The problem is formulated and solved using cvxpy, a python framework for 
convex programming. More information can be found in the paper CVXPY: A Python
-embedded modeling language for convex optimization by S. Diamond  and S. Boyd,
Journal of machine Learning Research (2016) (17,83 pp. 1 -5).

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

# np.random.seed(1)                                 # Activate line for reproducibility 


# ii) Definitions

n_meas_vecs=10          # nr of measurements to choose from
n_dim_theta=3           # nr of unknowns in the model A*theta=y
n_dim_x=n_meas_vecs     # nr of experiment densities to be determined



"""
    2. Randomly generate matrices --------------------------------------------
"""


# i) Measurement vectors 

V=np.random.normal(0,1,[n_meas_vecs,n_dim_theta])   # V @ theta =measurement
V_vecs=V.T                                          # cols are v_k



"""
    3. Formulate the semidefinite problem
"""


# i) Define variables

n_dim_opt_var=n_dim_x+n_dim_theta                    # nr of total optimization variables
opt_var=cp.Variable(n_dim_opt_var, nonneg=True)      # first n_dim_x : x, the rest: u


# ii) Define constraints

d=np.zeros([n_dim_opt_var])
d[0:n_dim_x]=1                  # constrain to relative frequency of measurements, i.e. 1^Tx=1
cons= [cp.scalar_product(d, opt_var)==1]

for k in range(n_dim_theta):
    LMI_mat=cp.bmat([[V.T@cp.diag(opt_var[0:n_dim_x])@V,np.reshape(np.eye(n_dim_theta)[:,k],[n_dim_theta,1])],
                     [np.reshape(np.eye(n_dim_theta)[:,k],[1,n_dim_theta]),cp.reshape(opt_var[n_dim_x+k],[1,1])]])
    cons=cons+[LMI_mat>>0]


# iii) Define objective function

c= np.zeros([n_dim_opt_var])
c[-n_dim_theta:n_dim_opt_var]=1

objective=cp.Minimize(cp.scalar_product(c, opt_var))



"""
    4. Assemble the solution -------------------------------------------------
"""


# i) Solve problem

prob=cp.Problem(objective,cons)
prob.solve(verbose=True)

# ii) Extract solution

x_opt=opt_var.value[0:n_dim_x]
u_opt=opt_var.value[n_dim_x:n_dim_opt_var]



"""
    5. Plots and illustratons ------------------------------------------------
"""


# i) Plot distribution of measurements

plt.figure(1,dpi=300)
plt.bar(np.linspace(1,n_dim_x,n_dim_x),x_opt,color='k')
plt.title('Optimal distribution of measurements')
plt.xlabel('Measurement nr')
plt.ylabel('Proportion of measurement')


# ii) Compare to other feasible choices of x (randomly chosen))

n_comparison=100
x_random=np.random.uniform(0,1,[n_dim_x,n_comparison])
x_sum=np.sum(x_random,axis=0)
x_normalize=np.repeat(np.reshape(1/x_sum,[1,n_comparison]),n_dim_x,axis=0)
x_random=x_random*x_normalize           # Normalize so that they sum to 1

obj_val=np.zeros([n_comparison])
for k in range(n_comparison):
    obj_val[k]=np.trace(np.linalg.pinv(V.T@np.diag(x_random[:,k])@V))
    
obj_val_optimal=np.trace(np.linalg.pinv(V.T@np.diag(x_opt)@V))
    
plt.figure(2,dpi=300)
plt.scatter(np.linspace(1,n_comparison,n_comparison),obj_val,color='k',label='Random x')
plt.scatter(0,obj_val_optimal,label='Optimal x',color='r')
plt.title('Uncertainties of experiment designs')
plt.ylabel('Trace of covariance matrix')
plt.xlabel('Experiment design nr.')
plt.legend()






























