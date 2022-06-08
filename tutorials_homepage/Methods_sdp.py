"""
The goal of this script is to solve a mini SDP illustrating bounding the maximal
eigenvalue of a covariance matrix.
For this, do the following:
    1. Definitions and imports
    2. Formulate the semidefinite problem
    3. Assemble the solution
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
import copy as copy
import matplotlib.pyplot as plt


# ii) Definitions



"""
    2. Formulate the semidefinite problem ------------------------------------
"""


# i) Define variables

n_dim_opt_var=2                                      # nr of total optimization variables
opt_var=cp.Variable(n_dim_opt_var)


# ii) Define constraints

LMI_mat=cp.bmat([[opt_var[0],opt_var[1]],[opt_var[1],1]])

cons=[LMI_mat>>0]+[opt_var[0]+opt_var[1]==2]


# iii) Define objective function

objective=cp.Minimize(cp.lambda_max(LMI_mat))



"""
    3. Assemble the solution -------------------------------------------------
"""


# i) Solve problem

prob=cp.Problem(objective,cons)
prob.solve(verbose=True)

# ii) Extract solution

x_opt=opt_var.value



"""
    4. Illustrate ------------------------------------------------------------
"""

# i) Create Grid for feasible set

n_grid=100
x=np.linspace(-3,3,n_grid)

xx1,xx2=np.meshgrid(x,x)
min_eigvals=np.zeros(np.shape(xx1))
max_eigvals=np.zeros(np.shape(xx1))

# ii) Calculate minimum eigenvalues

for k in range(n_grid):
    for l in range(n_grid):
        Mat_temp=np.array([[xx1[k,l],xx2[k,l]],[xx2[k,l],1]])
        min_eigvals[k,l]=np.min(np.linalg.eigvals(Mat_temp))
        max_eigvals[k,l]=np.max(np.linalg.eigvals(Mat_temp))
        
indicator_mat=copy.copy(min_eigvals)
indicator_mat[indicator_mat<=0]=0
indicator_mat[indicator_mat>0]=1        
        
        
# iii) Plot image

fig, ax = plt.subplots(1, 3, figsize=(10, 5),dpi=500)
ax[0].imshow(indicator_mat,extent=(-3,3,-3,3))
ax[0].set_title('Feasible set (yellow)')
ax[1].imshow(min_eigvals,extent=(-3,3,-3,3)) 
ax[1].set_title('Minimum eigenvalue')  
ax[2].imshow(max_eigvals,extent=(-3,3,-3,3))  
ax[2].set_title('Costs')








