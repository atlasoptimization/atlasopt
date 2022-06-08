"""
The goal of this script is to solve a mini SOCP illustrating a production plan
under stochastic constraints
For this, do the following:
    1. Definitions and imports
    2. Formulate the second order cone problem
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
from scipy.stats import norm


# ii) Definitions

c=np.array([[1],[2]])
a_bar=np.array([[0.7],[0.7]])
Sigma=0.1*np.eye(2)
Sigma_sqrt=np.sqrt(Sigma)                   # Works only for diagonal Sigma
b=2
eta=0.9



"""
    2. Formulate the second order cone problem --------------------------------
"""


# i) Define variables

n_dim_opt_var=[2,1]                        # nr of total optimization variables
opt_var=cp.Variable(n_dim_opt_var)


# ii) Define constraints

coeff=norm.ppf(eta)
cons=[coeff*cp.norm(Sigma_sqrt@opt_var)<=a_bar.T@opt_var-b]
cons=cons+[opt_var[0]>=0]
cons=cons+[opt_var[1]>=0]


# iii) Define objective function

objective=cp.Minimize(c.T@opt_var)



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
x=np.linspace(-1,5,n_grid)

xx1,xx2=np.meshgrid(x,x)
indicator_mat=np.zeros(np.shape(xx1))
cone_term=np.zeros(np.shape(xx1))
objective=np.zeros(np.shape(xx1))

# ii) Calculate minimum eigenvalues

for k in range(n_grid):
    for l in range(n_grid):
        x_vec_temp=np.array([[xx1[k,l]],[xx2[k,l]]])
        cone_term[k,l]=a_bar.T@x_vec_temp-b-coeff*np.linalg.norm(Sigma_sqrt@x_vec_temp)
        objective[k,l]=c.T@x_vec_temp
        
indicator_mat=copy.copy(cone_term)
indicator_mat[indicator_mat<=0]=0
indicator_mat[indicator_mat>0]=1     

indicator_mat=np.flipud(indicator_mat)
cone_term=np.flipud(cone_term)
objective=np.flipud(objective)   
        
        
# iii) Plot image

fig, ax = plt.subplots(1, 3, figsize=(10, 5),dpi=500)
ax[0].imshow(indicator_mat,extent=(-1,5,-1,5))
ax[0].set_title('Feasible set (yellow)')
ax[1].imshow(cone_term,extent=(-1,5,-1,5)) 
ax[1].set_title('Conic term')  
ax[2].imshow(objective,extent=(-1,5,-1,5))  
ax[2].set_title('Costs')






