"""
The goal of this script is to demonstrate policies as a result of optimal control.
The task is to guide a vehicle movable in the 2 D plane towards achieving a 
certain terminal location and velocity. The control input consists in accelerations 
leading to the state equation

x_k+1 = A x_k + B u_k
x_k=[position, velocity, acceleration] at timestep k
u_k= [acceleration] at timestep k

where A and B are matrices determining the state transitions. The control input
during the n timesteps is not known initially and the subject of the optimization
problem. Overall, we solve the minimization problem

min  sum_k |u_k|_1 
s.t. x_k+1=Ax_k+Bu_k
     x_1=x_start
     x_n=x_target  
     u_min<=u_k<=u_max
          
in the optimization variables x_k, k=1, ... ,n and u_k k=1, ... ,n. In the above, 
u_min and u_max are bounds on the control input and the terminal constraint
can involve positions, velocities and accelerations. We aim to achieve the 
terminal condition with as little control input as possible.
For this, do the following:
    1. Imports and definitions
    2. Generate matrices
    3. Formulate the optimization problem
    4. Assemble the solution
    5. Plots and illustratons
    
The discrete time unit is set to 1 for simplicity with the consequence of Delta t
not occuring anywhere in the script to convert acceleration to velocity or 
velocity to position.

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
n_dim=2                     # dimensionality
n_disc=5                    # nr of sample points for the policy

n_dim_u=n_dim                   # nr dim of one control vector
n_dim_x=n_dim*3                 # nr dim of one state
n_dim_all_u=n_dim*n_time        # nr of control variables
n_dim_all_x=n_dim_all_u*3           # number of trajectory variables


# iii) State transition matrices

zero_mat=np.zeros([n_dim,n_dim])
eye_mat=np.eye(n_dim)

A=np.bmat([[eye_mat, eye_mat, zero_mat],[zero_mat, eye_mat,eye_mat],[zero_mat,zero_mat,eye_mat]])
B=np.bmat([[zero_mat],[zero_mat],[eye_mat]])

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
    2. Generate matrices -----------------------------------------------------
"""


# i) Generate initial state

t=np.linspace(-50,50,n_disc)
xx1,xx2=np.meshgrid(t,t)
x_initial_full=np.vstack((xx1.flatten(),xx2.flatten(),np.zeros([2,n_disc**2]), 0.2*np.ones([2,n_disc**2])))


# ii) Constraints

x_terminal=np.zeros([6,1])
x_terminal=np.reshape(x_terminal,[n_dim_x,1])
u_min=-1*np.ones([n_dim_all_u,1])
u_max=1*np.ones([n_dim_all_u,1])



"""
    3. Formulate the optimization problem ------------------------------------
"""


# i) Define variables

x=cp.Variable((n_dim_all_x,1))
u=cp.Variable((n_dim_all_u,1))

x_opt=np.zeros([n_dim_all_x,n_disc**2])
u_opt=np.zeros([n_dim_all_u,n_disc**2])


# ii) Define constraints in loop

for k in range(n_disc**2):
        
    x_initial=np.reshape(x_initial_full[:,k],[n_dim_x,1])
    
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
    
    objective=cp.Minimize(cp.norm(u,p=1))           # Leads to sparse, sharp control
    # objective=cp.Minimize(cp.norm(u,p=2))         # leads to dense, smooth control

    
    
    """
        4. Assemble the solution ---------------------------------------------
    """
    
    
    # i) Solve problem 
    
    prob=cp.Problem(objective,cons)
    prob.solve(verbose=True)
    x_opt[:,k]=x.value.flatten()
    u_opt[:,k]=u.value.flatten()
    
    
# ii) Assemble solution

pos_opt_x=x_opt[0::6,:]
pos_opt_y=x_opt[1::6,:]
vel_opt_x=x_opt[2::6,:]
vel_opt_y=x_opt[3::6,:]
acc_opt_x=x_opt[4::6,:]
acc_opt_y=x_opt[5::6,:]

u_opt_x=u_opt[0::2]
u_opt_y=u_opt[1::2]



"""
    5. Plots and illustratons -------------------------------------------------
"""


# i) Illustration of optimal solution

plt.figure(1,dpi=300)
plt.plot(pos_opt_x,pos_opt_y,color='k')
plt.title('Optimal policy: positions')
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
# plt.scatter(x_initial[0],x_initial[1],color='k',label='Initial position')
plt.scatter(x_terminal[0],x_terminal[1],color='k',label='Terminal position')
plt.legend()

plt.figure(2,dpi=300)
plt.plot(vel_opt_x,vel_opt_y,color='k')
plt.title('Optimal policy: velocities')
plt.xlabel('Velocity in x direction')
plt.ylabel('Velocity in y direction')
plt.scatter(x_initial[2,0],x_initial[3,0],color='k',label='Initial velocity')
plt.scatter(x_terminal[2],x_terminal[3],color='k',label='Terminal velocity')

plt.figure(3,dpi=300)
plt.plot(acc_opt_x,acc_opt_y,color='k')
plt.title('Optimal policy: accelerations')
plt.xlabel('Acceleration in x direction')
plt.ylabel('Acceleration in y direction')
plt.scatter(x_initial[2,0],x_initial[3,0],color='k',label='Initial acceleration')
plt.scatter(x_terminal[2],x_terminal[3],color='k',label='Terminal acceleration')
































