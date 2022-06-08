"""
The goal of this script is to demonstrate robust control of a classic RLC circuit.
It features unknown parameters of capacitance and inductance that are bounded to
lie in a certain interval. The resistance is fixed. Instead of being able to observe
the full state x=[I,V] consisting in current and voltage, we only observe the 
the voltage and hove to decide upon an control signal based on that alone.
One gain matrix K facilitating state feedback control is to be designed
that stabilizes the system. The setting is that of gain scheduled output control,
i.e. we assume that the exact state transition matrix A is a superposition of
some basis A_i and the coefficients might be measured.
The control input consists in changes of I and V leading to the state equation

x_dot=Ax+Bu   
u=K y           y=Cx

y=system observations = [0,1]@x
x=[current, voltage] 
u=[change current, change voltage] 

where A and B are matrices determining the state transitions. A is bound to lie
in the convex hull (interval) between A_1, ..., A_4. The control input during the 
n timesteps is not known initially and the subject of the optimization problem. 
The goal is to find a matrix K with u=Ky stabilizing the system. Overall, we 
solve the feasibility problem

find  R, S
s.t. [S I]
     [I R] >>0
     N_R^T(A_i^T R + R A_i)N_R     i=1, ..., 4
     N_S^T(A_i^T S + S A_i)N_S     i=1, ..., 4


in the optimization variables R and S. K is then derivable from solving a second
feasibility problem of the form A_cl^TP+PA_cl<<0 for P constructed from R and S
for A_cl. After finding the state space matrices of a stable closed loop system 
in that way, the controller is assembled as a superposition of solutions. In 
essence this is a sequence of semidefinite feasibility problems.
For this, do the following:
    1. Imports and definitions
    2. Randomly generate data
    3. Formulate the optimization problems
    4. Assemble the solutions
    5. Plots and illustratons
    
The discrete time unit is set to 1 for simplicity with the consequence of Delta t
not occuring anywhere in the script to convert derivatives to state changes.

The specific RLC circuit described here is taken from pp. 10-11 Invitation to
Dynamical Systems by E.R. Scheinerman,  Dover publications (2012).
More Information on the SDP formulation be found e.g. in pp.428 - 431 Handbook 
of semidefinite  programming by H. Wolkowicz et. al., Springer Science & business 
Media (2003).
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
from scipy.linalg import null_space
import matplotlib.pyplot as plt

np.random.seed(0)                    # Activate line for reproducibility 


# ii) Definitions auxiliary


n_time=1000                  # nr of timesteps
n_dim=2                     # dimensionality

n_dim_x=n_dim                   # nr dim of one state
n_dim_xc=n_dim_x                # nr dim of completing state vector
t=np.linspace(0,1,n_time)
d_t=100/n_time


# iii) State transition matrices

R=0.1
L_1, L_2 = 1, 10         # inductance bounds
C_1, C_2 = 1, 10         # capacitance bounds

A_1=np.array([[0,-1/C_1],[1/L_1,-R/L_1]])
A_2=np.array([[0,-1/C_1],[1/L_2,-R/L_2]])
A_3=np.array([[0,-1/C_2],[1/L_1,-R/L_1]])
A_4=np.array([[0,-1/C_2],[1/L_2,-R/L_2]])


A_co=np.vstack([A_1.flatten(),A_2.flatten(),A_3.flatten(),A_4.flatten()]).T
A_collection=np.zeros([n_dim_x,n_dim_x,4])
for k in range(4):
    A_collection[:,:,k]=np.reshape(A_co[:,k],[n_dim_x,n_dim_x])

# iv) Matrices for the embedded closed loop

B_u=np.eye(2)
C_y=np.array([[1,0]])

n_dim_u=np.shape(B_u)[1]                    # nr dim of one control vector
n_dim_y=np.shape(C_y)[0]                    # nr dim of system output

A_0_tens=np.zeros([n_dim_x+n_dim_xc,n_dim_x+n_dim_xc,4])
for k in range(4):
    A_0_tens[:,:,k]=np.bmat([[np.reshape(A_co[:,k],[n_dim_x,n_dim_x]),np.zeros([n_dim_xc,n_dim_xc])],[np.zeros([n_dim_xc,n_dim_xc+n_dim_x])]])

B_full=np.bmat([[np.zeros([n_dim_x,n_dim_x]),B_u],[np.eye(n_dim_xc),np.zeros([n_dim_xc,n_dim_u])]])
C_full=np.bmat([[np.zeros([n_dim_xc,n_dim_x]),np.eye(n_dim_xc)],[C_y,np.zeros([n_dim_y,n_dim_xc])]])



"""
    2. Randomly generate data ------------------------------------------------
"""


# i) Generate initial state

mu=np.array([1,1])
Sigma=np.diag(np.array([0.1,0.1]))
x_initial=np.reshape(np.random.multivariate_normal(mu, Sigma),[n_dim_x,1])


# ii) Generate gain schedule

theta=np.zeros([4,n_time])
K_theta=np.zeros([n_time,n_time])
for k in range(n_time):
    for l in range(n_time):
        K_theta[k,l]=np.exp((-(t[k]-t[l])**2/0.2))
 
for k in range(4):
    theta[k,:]=np.random.multivariate_normal(np.zeros([n_time]), K_theta)

theta=np.abs(theta)
theta_sum=np.sum(theta,0)
for k in range(4):
    for l in range(n_time):
        theta[k,l]=theta[k,l]/theta_sum[l]    


# iii) Sample test run

x_test_run=np.zeros([n_dim_x,n_time+1])
x_test_run[:,0]=x_initial.flatten()
for k in range(n_time):
    A_theta=np.reshape(A_co@theta[:,k],[n_dim_x,n_dim_x])
    x_test_run[:,k+1]=(np.eye(n_dim_x)+d_t*A_theta)@x_test_run[:,k]



"""
    3. Formulate the optimization problems -----------------------------------
"""


# i) First feasibility problem

S=cp.Variable((n_dim_x,n_dim_x), PSD=True)
R=cp.Variable((n_dim_x,n_dim_x), PSD=True)

N_s=null_space(C_y)

cons=[]
for k in range(4):
    cons=cons+[N_s.T@(A_collection[:,:,k].T@S+S@A_collection[:,:,k])@N_s<<0]

cons=[cp.bmat([[S, np.eye(n_dim_x)],[np.eye(n_dim_x),R]])>>0]
    
objective=cp.Minimize(cp.trace(R)+cp.trace(S))
prob=cp.Problem(objective,cons)
prob.solve(cp.SCS, verbose=True)


# ii) Assemble P enabling feasible A_cl

R_val=R.value
S_val=S.value

P_temp=(S_val-np.linalg.pinv(R_val))
U_p,S_p,V_p=np.linalg.svd(P_temp)
P_12=U_p@np.diag(np.sqrt(S_p))@V_p.T
Q_12=-R_val@P_12

P_mat=np.bmat([[S_val, P_12],[P_12.T,np.eye(n_dim_x)]])
Q_mat=np.linalg.pinv(P_mat)


# iii) Sequence of feasibility problems on vertex points -setup

Omega_1=cp.Variable((n_dim_xc+n_dim_u,n_dim_xc+n_dim_y))
cons_1=[(A_0_tens[:,:,0]+B_full@Omega_1@C_full).T@P_mat+P_mat@(A_0_tens[:,:,0]+B_full@Omega_1@C_full)<<0]

Omega_2=cp.Variable((n_dim_xc+n_dim_u,n_dim_xc+n_dim_y))
cons_2=[(A_0_tens[:,:,1]+B_full@Omega_2@C_full).T@P_mat+P_mat@(A_0_tens[:,:,1]+B_full@Omega_2@C_full)<<0]

Omega_3=cp.Variable((n_dim_xc+n_dim_u,n_dim_xc+n_dim_y))
cons_3=[(A_0_tens[:,:,2]+B_full@Omega_3@C_full).T@P_mat+P_mat@(A_0_tens[:,:,2]+B_full@Omega_3@C_full)<<0]

Omega_4=cp.Variable((n_dim_xc+n_dim_u,n_dim_xc+n_dim_y))
cons_4=[(A_0_tens[:,:,3]+B_full@Omega_4@C_full).T@P_mat+P_mat@(A_0_tens[:,:,3]+B_full@Omega_4@C_full)<<0]



"""
    4. Assemble the solutions ------------------------------------------------
"""


# i) Sequence of feasibility problems on vertex points- solution

objective_1=cp.Minimize(cp.sum_squares(Omega_1))
prob_1=cp.Problem(objective_1,cons_1)
prob_1.solve(cp.SCS,verbose=True)
Omega_1_val=Omega_1.value

objective_2=cp.Minimize(cp.sum_squares(Omega_2))
prob_2=cp.Problem(objective_2,cons_2)
prob_2.solve(cp.SCS,verbose=True)
Omega_2_val=Omega_2.value

objective_3=cp.Minimize(cp.sum_squares(Omega_3))
prob_3=cp.Problem(objective_3,cons_3)
prob_3.solve(cp.SCS,verbose=True)
Omega_3_val=Omega_3.value

objective_4=cp.Minimize(cp.sum_squares(Omega_4))
prob_4=cp.Problem(objective_4,cons_4)
prob_4.solve(cp.SCS,verbose=True)
Omega_4_val=Omega_4.value


# ii) Assemble solution - closed loop A

A_cl=np.zeros([n_dim_x+n_dim_xc,n_dim_x+n_dim_xc,4])
Omega=np.zeros([n_dim_xc+n_dim_u,n_dim_xc+n_dim_y,4])
Omega[:,:,0]=Omega_1_val
Omega[:,:,1]=Omega_2_val
Omega[:,:,2]=Omega_3_val
Omega[:,:,3]=Omega_4_val


for k in range(4):
    A_cl[:,:,k]=A_0_tens[:,:,k]+B_full@Omega[:,:,k]@C_full




"""
    5. Plots and illustratons ------------------------------------------------
"""


# i) Illustration of optimal solution and some simple guessed alternative

xc_opt_run=np.zeros([n_dim_x+n_dim_xc,n_time+1])
xc_opt_run[:,0]=np.hstack((x_initial.flatten(),np.zeros([n_dim_xc])))
x_alt_run=np.zeros([n_dim_x,n_time+1])
x_alt_run[:,0]=x_initial.flatten()
K_y_alt=-0.85*np.ones([n_dim_u,1])


for k in range(n_time):
    A_theta=np.reshape(A_co@theta[:,k],[n_dim_x,n_dim_x])
    x_alt_run[:,k+1]=(np.eye(n_dim_x))@x_alt_run[:,k]+d_t*(A_theta+K_y_alt@C_y)@x_alt_run[:,k]
    
    A_cl_theta=np.reshape(A_cl.reshape([(n_dim_x+n_dim_xc)**2,4])@theta[:,k],[n_dim_x+n_dim_xc,n_dim_x+n_dim_xc])
    xc_opt_run[:,k+1]=(np.eye(n_dim_x+n_dim_xc))@xc_opt_run[:,k]+d_t*(A_cl_theta)@xc_opt_run[:,k]
    
x_opt_run=xc_opt_run[0:2,:]
    

w,h=plt.figaspect(0.2)
fig1 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs1 = fig1.add_gridspec(1, 4)

f1_ax1 = fig1.add_subplot(gs1[0,0])
f1_ax1.plot(theta[0,:], theta[1,:],color='k')
plt.title(r'Random $\theta$')
plt.xlabel(r'$\theta 1$')
plt.ylabel(r'$\theta 2$')

f1_ax2 = fig1.add_subplot(gs1[0,1])
f1_ax2.plot(x_opt_run[0,:], x_opt_run[1,:],color='k')
plt.title(r'Optimally stabilized circuit for random $\theta$ ')
plt.xlabel('Current I')
plt.ylabel('Voltage V')

f1_ax3 = fig1.add_subplot(gs1[0,2])
f1_ax3.plot(x_alt_run[0,:], x_alt_run[1,:],color='k')
plt.title('Circuit with freely guessed control law  ')
plt.xlabel('Current I')
plt.ylabel('Voltage V')

f1_ax4 = fig1.add_subplot(gs1[0,3])
f1_ax4.plot(x_test_run[0,:], x_test_run[1,:],color='k')
plt.title('Circuit without control law')
plt.xlabel('Current I')
plt.ylabel('Voltage V')





























