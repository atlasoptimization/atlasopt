"""
The goal of this script is to demonstrate robust control of a harmonic oscillator
whose precise resonance frequency is unknown but bounded to lie in a certain
interval. One gain matrix K facilitating state feedback control is to be designed
that stabilizes the oscillator regardless of the precise resonance frequency.
The control input consists in accelerations leading to the state equation

x_k+1 =x_k+dt( A x_k + B u_k)
x_k=[position, velocity] at timestep k
u_k= [acceleration] at timestep k

where A and B are matrices determining the state transitions. A is bound to lie
in the convex hull (interval) between A_1 and A_2. The control input during the 
n timesteps is not known initially and the subject of the optimization problem. 
The goal is to find a matrix K with u=Kx stabilizing the system Overall, we 
solve the minimization problem

min  trace(Q)
s.t. Q > 0
     (Q A_1^T+A_1Q)+(Y^TB^T+BY)
     (Q A_2^T+A_2Q)+(Y^TB^T+BY)


in the optimization variables Q and Y. K is then given by K=YQ^{-1}. In essence
this is a semidefinte feasibility problem, on which we have imposed the entries
 of involved matrices to be reasonably small.
For this, do the following:
    1. Imports and definitions
    2. Randomly generate matrices
    3. Formulate the optimization problem
    4. Assemble the solution
    5. Plots and illustratons
    
The discrete time unit is set to 1 for simplicity with the consequence of Delta t
not occuring anywhere in the script to convert acceleration to velocity or 
velocity to position.

More Information can be found e.g. in pp.428 - 431 Handbook of semidefinite 
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
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


# ii) Definitions auxiliary


n_time=1000                 # nr of timesteps
n_dim=1                     # dimensionality

n_dim_u=n_dim                   # nr dim of one control vector
n_dim_x=n_dim*2                 # nr dim of one state
t=np.linspace(0,1,n_time)
d_t=20/n_time


# iii) State transition matrices

w_1=0.1                         # Square of low resonance frequency
w_2=5                         # Square of high resonance frequency
w_mean=1                        # nominal simple task

A_1=np.array([[0,1],[-w_1,0]])
A_2=np.array([[0,1],[-w_2,0]])
A_mean=np.array([[0,1],[-w_mean,0]])
B=np.array([[0],[1]])



"""
    2. Randomly generate matrices ---------------------------------------------
"""


# i) Generate initial state

np.random.seed(0)
mu=np.array([-0,0])
Sigma=np.diag(np.array([1,0]))
x_initial=np.reshape(np.random.multivariate_normal(mu, Sigma),[n_dim_x,1])


# ii) Sample test run

x_test_run_1=np.zeros([n_dim_x,n_time+1])
x_test_run_1[:,0]=x_initial.flatten()
x_test_run_mean=np.zeros([n_dim_x,n_time+1])
x_test_run_mean[:,0]=x_initial.flatten()
x_test_run_2=np.zeros([n_dim_x,n_time+1])
x_test_run_2[:,0]=x_initial.flatten()

for k in range(n_time):
    x_test_run_1[:,k+1]=(np.eye(n_dim_x)+d_t*A_1)@x_test_run_1[:,k]
    x_test_run_mean[:,k+1]=(np.eye(n_dim_x)+d_t*A_mean)@x_test_run_mean[:,k]
    x_test_run_2[:,k+1]=(np.eye(n_dim_x)+d_t*A_2)@x_test_run_2[:,k]


"""
    3. Formulate the semidefinite problem ------------------------------------
"""


# i) Define variables

Q=cp.Variable((n_dim_x,n_dim_x),PSD=True)
Y=cp.Variable((n_dim_u,n_dim_x))


# ii) Define constraints

cons=[cp.trace(Q)==1]
cons=cons+[-Q@A_1.T-A_1@Q-Y.T@B.T-B@Y>>0]
cons=cons+[-Q@A_2.T-A_2@Q-Y.T@B.T-B@Y>>0]


# iii) Define objective function

objective=cp.Minimize(cp.trace(Q))           



"""
    4. Assemble the solution -------------------------------------------------
"""


# i) Solve problem 

prob=cp.Problem(objective,cons)
prob.solve(cp.SCS, verbose=True)
Y_opt=Y.value
Q_opt=Q.value


# ii) Assemble solution

K_opt=Y_opt@np.linalg.pinv(Q_opt)
# K_opt=np.array([[0,0]])
# K_opt=np.array([[0,-0.5]])

# Check eigenvals
print(np.linalg.eig(-Q_opt@A_1.T-A_1@Q_opt-Y_opt.T@B.T-B@Y_opt))
print(np.linalg.eig(-Q_opt@A_2.T-A_2@Q_opt-Y_opt.T@B.T-B@Y_opt))



"""
    5. Plots and illustratons -----------------------------------------------
"""


# i) Illustration of optimal solution

x_opt_run_1=np.zeros([n_dim_x,n_time+1])
x_opt_run_1[:,0]=x_initial.flatten()
x_opt_run_2=np.zeros([n_dim_x,n_time+1])
x_opt_run_2[:,0]=x_initial.flatten()
x_opt_run_mean=np.zeros([n_dim_x,n_time+1])
x_opt_run_mean[:,0]=x_initial.flatten()

for k in range(n_time):
    x_opt_run_1[:,k+1]=(np.eye(n_dim_x))@x_opt_run_1[:,k]+d_t*(A_1+B@K_opt)@x_opt_run_1[:,k]
    x_opt_run_2[:,k+1]=(np.eye(n_dim_x))@x_opt_run_2[:,k]+d_t*(A_2+B@K_opt)@x_opt_run_2[:,k]
    x_opt_run_mean[:,k+1]=(np.eye(n_dim_x))@x_opt_run_mean[:,k]+d_t*(A_mean+B@K_opt)@x_opt_run_mean[:,k]

w,h=plt.figaspect(0.3)
fig1 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs1 = fig1.add_gridspec(1, 3)

f1_ax1 = fig1.add_subplot(gs1[0,0])
f1_ax1.plot(t, x_opt_run_1[0,0:-1],color='k')
plt.title('Stabilized oscillator for $\omega$ = ' '{0:.2f}'.format(w_1))
plt.xlabel('time t')
plt.ylabel('function value')

f1_ax2 = fig1.add_subplot(gs1[0,1])
f1_ax2.plot(t, x_opt_run_mean[0,0:-1],color='k')
plt.title('Stabilized oscillator for $\omega$ = ' '{0:.2f}'.format(w_mean))
plt.xlabel('time t')
plt.ylabel('function value')

f1_ax3 = fig1.add_subplot(gs1[0,2])
f1_ax3.plot(t, x_opt_run_2[0,0:-1],color='k')
plt.title('Stabilized oscillator for $\omega$ = ' '{0:.2f}'.format(w_2))
plt.xlabel('time t')
plt.ylabel('function value')


# ii) Illustration of uncontrolled behavior

w,h=plt.figaspect(0.3)
fig2 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs2 = fig1.add_gridspec(1, 3)

f2_ax1 = fig2.add_subplot(gs2[0,0])
f2_ax1.plot(t, x_test_run_1[0,0:-1],color='k')
plt.title('Unperturbed oscillator for $\omega$ = ' '{0:.2f}'.format(w_1))
plt.xlabel('time t')
plt.ylabel('function value')

f2_ax2 = fig2.add_subplot(gs2[0,1])
f2_ax2.plot(t, x_test_run_mean[0,0:-1],color='k')
plt.title('Unperturbed oscillator for $\omega$ = ' '{0:.2f}'.format(w_mean))
plt.xlabel('time t')
plt.ylabel('function value')

f2_ax3 = fig2.add_subplot(gs2[0,2])
f2_ax3.plot(t, x_test_run_2[0,0:-1],color='k')
plt.title('Unperturbed oscillator for $\omega$ = ' '{0:.2f}'.format(w_2))
plt.xlabel('time t')
plt.ylabel('function value')





























