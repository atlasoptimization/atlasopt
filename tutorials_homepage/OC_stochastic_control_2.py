"""
The goal of this script is to demonstrate optimal control of a system subjected
to random influences. The state transition dynamics are assumed known; the 
system illustrated here represents a rigid pendulum whose angular accelerations
can be changed via the control variable and are subject to random influences.
The discrete state transition can roughly be written as

(d/dt)x_k = A x_k + B u_k + w_k
x_k=[phi, (d/dt) phi, (d/dt)^2 phi, sin phi , cos phi] at timestep k
u_k= [angular acceleration] at timestep k
w_k=[random angular acceleration] at timestep k

where A and B are matrices determining the state transitions and (d/dt) denotes
time derivatives,. The control input during the n timesteps is not known 
initially and the subject of the optimization problem. Overall, we solve the 
minimization problem

min  sum_k Expected[ x_k^TQx_k + u_k^TRu_k]
s.t. (d/dt)x_k=A x_k +B u_k +w_k
          ]
in the optimization variables_k k=1, ... ,n. This is achievable via a a constant
Feedback u=Kx with K a matrix determined by solving the following problem.

min  tr(Q Z_xx) + tr(R Z_uu)
s.t. [Z_xx   Z_xu]
     [Z_xu^T Z_uu]  >>0
     Z_xx-A Z_xx A^T - A Z_xuB^T -B Z_xu^T A^T - BZ_uuB^T=W

Any solution to the above semidefinite program yields the correlations between
states and input signals that minimize the sum of squared costs. The gain matrix
K = Z_xu^T(Z_uu)^{-1} immediately follows and u=Kx solves the optimal control
problem.
For this, do the following:
    1. Imports and definitions
    2. Generate matrices
    3. Formulate the optimization problem
    4. Assemble the solution
    5. Plots and illustratons
    
The discrete time unit is set to 0.01 for simplicity with the consequence of Delta t
not occuring anywhere in the script to convert acceleration to velocity or 
velocity to position.

Linear quadratic Gaussian control is an important problem. A derivation of the \
optimization formulations can be found in the paper On infinite dimensional linear
programming approach to stochastic control by M. Kamgarpour and T. Summers, arXiv
(2018).
The problem is formulated and solved using cvxpy, a python framework for 
convex programming. More information can be found in the paper CVXPY: A Python
-embedded modeling language for convex optimization by S. Diamond  and S. Boyd,
Journal of machine Learning Research (2016) (17,83 pp. 1 -5).

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""



"""
    1. Imports and definitions ------------------------------------------------
"""


# i) Imports

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# np.random.seed(1)                         # Activate line for reproducibility 
w_var=0.003                                 # Increae for more randomness

# ii) Definitions auxiliary

n_time=10000              # nr of timesteps
n_dim_x=5               # nr of dim state
n_dim_u=1               # nr of dim control
n_dim_w=5               # nr of dim noise


# iii) State transition matrices

g=10
Delta_t=0.001
t=np.linspace(0,Delta_t*n_time,n_time)

# Small angle approximation: sin phi+Delta_phi = sin phi + Delta_t *(d/dt)phi
A=np.array([[1,Delta_t,0,0,0],[0,1,Delta_t,0,0],[0,0,0,-g,0],[0,Delta_t,0,1,0],[0,0,0,0,1]]) 
B=np.array([[0],[0],[1],[0],[0]])

Q=np.eye(n_dim_x); Q[3,3]=0; Q[4,4]=0
R=np.eye(n_dim_u)




"""
    2. Generate matrices -----------------------------------------------------
"""


# i) Generate initial state

W=np.zeros([n_dim_x,n_dim_x])
W[2,2]=w_var

w=np.zeros([n_dim_x,n_time])
w[2,:]=np.random.normal(0,w_var,[1,n_time])

u_samplerun=np.zeros([n_dim_u,n_time])


# ii) Sample trivial run

x_samplerun=np.zeros([n_dim_x,n_time])
x_samplerun[:,0]=np.random.normal(0,0.03,[n_dim_x])
x_samplerun[3,0]=np.sin(x_samplerun[0,0]); x_samplerun[4,0]=np.cos(x_samplerun[0,0])
x_samplerun[2,0]=-g*np.sin(x_samplerun[0,0])
for k in range(n_time-1):
        x_samplerun[:,k+1]=A@x_samplerun[:,k]+B@u_samplerun[:,k]+w[:,k]



"""
    3. Formulate the semidefinite problem ------------------------------------
"""


# i) Define variables

Z_xx=cp.Variable([n_dim_x,n_dim_x])
Z_uu=cp.Variable([n_dim_u,n_dim_u])
Z_xu=cp.Variable([n_dim_x,n_dim_u])



# ii) Define constraints

cons=[cp.bmat([[Z_xx,Z_xu],[Z_xu.T,Z_uu]])>>0]
cons=cons+[W+A@Z_xx@A.T+A@Z_xu@B.T+B@Z_xu.T@A.T+B@Z_uu@B.T==Z_xx]
               
               

"""
    4. Assemble the solution -------------------------------------------------
"""


# i) Solve problem 

obj_fun=cp.Minimize(cp.trace(Q@Z_xx)+cp.trace(R@Z_uu))
prob=cp.Problem(obj_fun,cons)
prob.solve(verbose=True)


# ii) Assemble solution

Z_xu_val=Z_xu.value
Z_xx_val=Z_xx.value

K=Z_xu_val.T@np.linalg.pinv(Z_xx_val)


# iii) Run a solution

x_optirun=np.zeros([n_dim_x,n_time])
x_optirun[:,0]=x_samplerun[:,0]
u_optirun=np.zeros([n_dim_u,n_time])
for k in range(n_time-1):
    u_optirun[:,k]=K@x_optirun[:,k]
    x_optirun[:,k+1]=A@x_optirun[:,k]+B@u_optirun[:,k]+w[:,k]



"""
    5. Plots and illustratons -------------------------------------------------
"""


# i) Illustration of optimal solution

fig1 = plt.figure(dpi=200,constrained_layout=True)
gs1 = fig1.add_gridspec(3, 1)
f1_ax1 = fig1.add_subplot(gs1[0,0])
f1_ax1.plot(t,x_samplerun[0,:],'k')
f1_ax1.set_title('$\phi$')

f1_ax2 = fig1.add_subplot(gs1[1,0])
f1_ax2.plot(t,x_samplerun[1,:], 'k')
f1_ax2.set_title('$(d/dt) \phi$')

f1_ax3 = fig1.add_subplot(gs1[2,0])
f1_ax3.plot(t,x_samplerun[2,:], 'k')
f1_ax3.set_title('$(d/dt)^2 \phi$')

f2 = plt.figure(dpi=200,constrained_layout=True)
f2_ax1 = plt.axes(projection='3d')
f2_ax1.plot3D(x_samplerun[0,:], x_samplerun[1,:], x_samplerun[2,:], 'black')
f2_ax1.set_title('3D trajectory of uncontrolled states')
f2_ax1.set_xlabel('$\phi$ axis')
f2_ax1.set_ylabel('$(d/dt) \phi$ axis')
f2_ax1.set_zlabel('$(d/dt)^2 \phi$ axis')

fig3 = plt.figure(dpi=200,constrained_layout=True)
gs3 = fig3.add_gridspec(4, 1)
f3_ax1 = fig3.add_subplot(gs3[0,0])
f3_ax1.plot(t,x_optirun[0,:], 'k')
f3_ax1.set_title('$\phi$')

f3_ax2 = fig3.add_subplot(gs3[1,0])
f3_ax2.plot(t,x_optirun[1,:], 'k')
f3_ax2.set_title('$(d/dt) \phi$')

f3_ax3 = fig3.add_subplot(gs3[2,0])
f3_ax3.plot(t,x_optirun[2,:], 'k')
f3_ax3.set_title(' $(d/dt)^2 \phi$')

f3_ax4 = fig3.add_subplot(gs3[3,0])
f3_ax4.plot(t,u_optirun[0,:], 'k')
f3_ax4.set_title('Control signal')

f4 = plt.figure(dpi=200,constrained_layout=True)
f4_ax1 = plt.axes(projection='3d')
f4_ax1.plot3D(x_optirun[0,:], x_optirun[1,:], x_optirun[2,:], 'black')
f4_ax1.set_title('3D trajectory of controlled states')
f4_ax1.set_xlabel('$\phi$ axis')
f4_ax1.set_ylabel('$(d/dt) \phi$ axis')
f4_ax1.set_zlabel('$(d/dt)^2 \phi$ axis')





















