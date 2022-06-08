"""
The goal of this script is to illustrate the damped harmonic oscillators behavior
that can be manipulated by a control signal.
For this, do the following:
    1. Definitions and imports
    2. Simulate sequentially
    3. Check controllability and observability
    4. Optimally control system evolution
    5. Plots and illustrations

The differential equation governing the system is x_dot=Ax+Bu with x=[position,
velocity] where x_dot is the time derivative of the state, u is the control 
signal, and A and B are matrices mapping state and control signal to state change.

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""


"""
    1. Definitions and imports -----------------------------------------------
"""


# i) Imports

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# ii) Definitions

n=5000
Delta_t=0.01
t=np.linspace(0,n*Delta_t,n)



"""
    2. Simulate sequentially -------------------------------------------------
"""


# i) Set up Matrices

k=1; m=1; c=0.2

c_1=c/m
c_2=k/m

A=np.array([[0,1],[-c_2,-c_1]])
B=np.array([[0],[1]])
C=np.eye(2)

# ii) Run a simulation

x_full=np.zeros([2,n])
x_full[0,0]=1

for k in range(1,n):
    x_full[:,k]=x_full[:,k-1]+Delta_t*A@x_full[:,k-1]
    
    
    
"""
    3. Check controllability and observability
"""


# i) Controllability SDP

P_c=cp.Variable((2,2),symmetric=True)
cons_c=[P_c@B==np.zeros([2,2])]
cons_c=cons_c+[A.T@P_c+P_c@A<<0]

objective_c=cp.Minimize(cp.trace(-P_c))
problem_controllability=cp.Problem(objective_c,cons_c)
problem_controllability.solve(verbose=True)

# If the problem is infeasible or only has solution P_c==0, the the system is controllable.
if problem_controllability.status!='optimal' or np.all(np.isclose(P_c.value,0)):
    control_check=True
else:
    control_check=False


# ii) Observability SDP

P_o=cp.Variable((2,2),symmetric=True)
cons_o=[P_o@C.T==np.zeros([2,2])]
cons_o=cons_o+[A@P_o+P_o@A.T<<0]

objective_o=cp.Minimize(cp.trace(-P_o))
problem_observability=cp.Problem(objective_o,cons_o)
problem_observability.solve(verbose=True)

# If the problem is infeasible or only has solution P_c==0, the the system is controllable.
if problem_observability.status!='optimal' or np.all(np.isclose(P_o.value,0)):
    observe_check=True
else:
    observe_check=False



"""
    4. Optimally control system evolution
"""


# i) Pose optimization problem for P

R=np.eye(1)
Q=np.eye(2)

P=cp.Variable((2,2),symmetric=True)
cons=[cp.bmat([[R, B.T@P],[P@B, Q+A.T@P+P@A]])>>0]

objective=cp.Minimize(cp.trace(-P))
prob=cp.Problem(objective,cons)


# ii) Solve optimization problem for P and design K

prob.solve(verbose=True)
P_control=P.value

K=-np.linalg.pinv(R)@B.T@P_control


# iii) Run a simulation under control laws

x_full_control=np.zeros([2,n])
x_full_control[0,0]=1
u=np.zeros([1,n])

for k in range(1,n):
    u[0,k-1]=K@x_full_control[:,k-1]
    x_full_control[:,k]=x_full_control[:,k-1]+Delta_t*(A@x_full_control[:,k-1]+B@K@x_full_control[:,k-1])




"""
    5. Plots and illustrations -----------------------------------------------
"""


# i) Figure on temporal behavior uncontrolled

plt.figure(1,dpi=300)
plt.plot(t,x_full[0,:],color='k')
plt.title('Temporal behavior for uncontrolled system')
plt.xlabel('Time t')
plt.ylabel('Position x')


# ii) Figure on phasespace behavior uncontrolled

plt.figure(2,dpi=300)
plt.plot(x_full[0,:],x_full[1,:],color='k')
plt.title('Temporal behavior for uncontrolled system')
plt.xlabel('Position x')
plt.ylabel('Velocity $\dot{x}$')


# iii) Figure on temporal behavior controlled

plt.figure(3,dpi=300)
plt.plot(t,x_full_control[0,:],color='k', label='state')
plt.plot(t,u[0,:],color='k', linestyle='--', label='control signal')
plt.title('Temporal behavior for controlled system')
plt.xlabel('Time t')
plt.ylabel('Position x')
plt.legend()


# iv) Figure on phasespace behavior controlled

plt.figure(4,dpi=300)
plt.plot(x_full_control[0,:],x_full_control[1,:],color='k')
plt.title('Temporal behavior for controlled system')
plt.xlabel('Position x')
plt.ylabel('Velocity $\dot{x}$')




