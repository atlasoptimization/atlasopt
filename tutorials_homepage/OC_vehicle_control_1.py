"""
The goal of this script is to demonstrate optimal control of an RC car that is to
be guided to a specific location. The control input consists in accelerations 
leading to the state equation

x_k+1 = A x_k + B u_k
x_k=[position, velocity, acceleration] at timestep k
u_k= [acceleration] at timestep k

where A and B are matrices determining the state transitions. The control input
during the n timesteps is not known initially and the subject of the optimization
problem. Overall, we solve the minimization problem

min  sum_k (u_k)^2 
s.t. x_k+1=Ax_k+Bu_k
     x_1=[0,0,0], the initial state
     x_n=[1,0,0], the target state
     u_min<=u_k<=u_max
          
in the optimization variables x_k, k=1, ... ,n and u_k k=1, ... ,n. In the above, 
u_min and u_max are bounds on the control input and the terminal constraint
determines the end position. We aim to achieve the terminal condition with as 
little expended energy as possible.
For this, do the following:
    1. Imports and definitions
    2. Generate matrices
    3. Formulate the optimization problem
    4. Assemble the solution
    5. Plots and illustratons
    
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
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


# ii) Definitions auxiliary

n_time=100                  # nr of timesteps

n_dim_u=1                   # nr dim of one control vector
n_dim_x=3                   # nr dim of one state
n_dim_all_u=n_time            # nr of control variables
n_dim_all_x=n_dim_all_u*3           # number of trajectory variables



"""
    2. Generate matrices -----------------------------------------------------
"""


# i) Generate initial state

x_initial=np.zeros([n_dim_x,1])


# ii) Constraints

x_terminal=np.array([[1],[0],[0]])
u_min=-1*np.ones([n_dim_all_u,1])
u_max=1*np.ones([n_dim_all_u,1])


# iii) State transition matrices

Delta_t=0.1
t=np.linspace(0,Delta_t*n_time,n_time)

A=np.array([[1,Delta_t,0],[0,1,Delta_t],[0,0,1]])
B=np.array([[0],[0],[1]])

# High dimensional form s.t. x-A_full x -B_full u=0
A_full=A
for k in range(n_time-2):
    A_full=block_diag(A_full,A)
A_full=np.bmat([[np.zeros([n_dim_x,n_dim_all_x-n_dim_x]),np.zeros([n_dim_x,n_dim_x])],[A_full,np.zeros([n_dim_all_x-n_dim_x,n_dim_x])]])
A_full[0:n_dim_x,0:n_dim_x]=np.eye(n_dim_x)

B_full=B
for k in range(n_time-2):
    B_full=block_diag(B_full,B)
B_full=np.bmat([[np.zeros([n_dim_x,n_dim_all_u-n_dim_u]),np.zeros([n_dim_x,n_dim_u])],[B_full,np.zeros([n_dim_all_x-n_dim_x,n_dim_u])]])




"""
    3. Formulate the optimization problem ------------------------------------
"""


# i) Define variables

x=cp.Variable((n_dim_all_x,1))
u=cp.Variable((n_dim_all_u,1))


# ii) Define constraints

# Direct constraints on x
cons=[]
cons=cons+[x[0:n_dim_x]==x_initial]
cons=cons+[x[n_dim_all_x-n_dim_x:n_dim_all_x]==x_terminal]

# Bounds on u
cons=cons+[u<=u_max]
cons=cons+[u>=u_min]

# Dynamic constraints
cons=cons+[x-A_full@x-B_full@u==np.zeros([n_dim_all_x,1])]


# iii) Define objective function

objective=cp.Minimize(cp.norm(x[2::3],p=2))           # Leads to sparse, sharp control (Penalize acceleration)
# objective=cp.Minimize(cp.norm(u,p=2))         # leads to dense, smooth control (Penalize acceleration change)



"""
    4. Assemble the solution --------------------------------------------------
"""


# i) Solve problem 

prob=cp.Problem(objective,cons)
prob.solve(verbose=True)
x_opt=x.value
u_opt=u.value


# ii) Assemble solution

pos_opt=x_opt[0::3]
vel_opt=x_opt[1::3]
acc_opt=x_opt[2::3]

u_opt=u_opt[:]



"""
    5. Plots and illustratons ------------------------------------------------
"""


# i) Illustration of optimal solution

plt.figure(1,dpi=300)
plt.plot(t,pos_opt,color='k')
plt.title('Optimal trajectory: position $x(t)$')
plt.xlabel('Time t')
plt.ylabel('x coordinate')
plt.scatter(t[0],x_initial[1],color='k',label='Initial position')
plt.scatter(t[-1],x_terminal[0],color='k',label='Terminal position')
plt.legend()

plt.figure(2,dpi=300)
plt.plot(t,vel_opt,color='k')
plt.title('Optimal trajectory: velocity $\dot{x}(t)$')
plt.xlabel('Time t')
plt.ylabel('Velocity')
plt.scatter(t[0],x_initial[1],color='k',label='Initial velocity')
plt.scatter(t[-1],x_terminal[1],color='k',label='Terminal velocity')
plt.legend()

plt.figure(3,dpi=300)
plt.plot(t,acc_opt,color='k')
plt.title('Optimal trajectory: acceleration $\ddot{x}(t)$')
plt.xlabel('Time t')
plt.ylabel('Acceleration')
plt.scatter(t[0],x_initial[2],color='k',label='Initial acceleration')
plt.scatter(t[-1],x_terminal[2],color='k',label='Terminal acceleration')
plt.legend()

plt.figure(4,dpi=300)
plt.plot(t,u_opt,color='k')
plt.title('Optimal trajectory: control inputs')
plt.xlabel('Time t')
plt.ylabel('Control acceleration')
































