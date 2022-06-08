""" 
The goal of this script is to solve a mini MILP showing how to include disjunctive 
(OR) constraints. The optimization problem is
    min_x c_1x_1+c_2x_2 
    s.t.  x_1 >=0, x_2>=0 
          x_1+x_2=2
          and either x_1 <=1.5 or x_2 >=1.5
We model this disjunctive constraint of requiring one (or both) of these constraints
to be true as a mixed integer linear program.
For this, do the following:
    1. Definitions and imports
    2. Formulate the mixed integer problem
    3. Assemble the solution
    4. Illustrate

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""



"""
    1. Imports and definitions ------------------------------------------------
"""


# i) Imports

import numpy as np
import cvxpy as cp
import time


# ii) Definitions

M1=100
M2=100
c=np.array([[1],[2]])



"""
    2. Formulate the mixed integer problem -----------------------------------
"""


# i) Define variables

n_dim_opt_var_x=2                                      # nr of x optimization variables
n_dim_opt_var_y=2                                      # nr of x optimization variables
opt_var_x=cp.Variable(n_dim_opt_var_x,nonneg=True)
opt_var_y=cp.Variable(n_dim_opt_var_y,boolean=True)

# ii) Define constraints

cons=[opt_var_x[0]+opt_var_x[1]==2]             # Simple equality constraint
cons=cons+[opt_var_y[0]+opt_var_y[1]==1]        # Penalty for one disjunctive constraint active
cons=cons+[opt_var_x[0]-M1*opt_var_y[0]<=1.5]   # Constraint 1
cons=cons+[-opt_var_x[1]-M2*opt_var_y[1]<=-1.5] # Constraint 2


# iii) Define objective function

objective=cp.Minimize(c.T@opt_var_x)



"""
    3. Assemble the solution -------------------------------------------------
"""


# i) Solve problem

prob=cp.Problem(objective,cons)
prob.solve(verbose=True)


# ii) Extract solution

x_opt=opt_var_x.value
y_opt=opt_var_y.value

time.sleep(0.01)


"""
    4. Illustrate-------------------------------------------------------------
"""


# i) Create Grid for feasible set

print('The optimal choice for the vector x is',x_opt)
print('The optimal choice for the logical vector y is',y_opt)
print('The position of the 0 in y indicates the constraint that is satisfied')








