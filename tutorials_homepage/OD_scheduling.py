"""
The goal of this script is to demonstrate optimal scheduling. The goal is to derive 
a schedule that minimizes total completion time. The problem features 12 jobs to
be scheduled on 3 machines with different processing times. It is assumed that 
ressource consumption is the same for all machines and processes thereby making 
the problem into an LP. We solve this LP and interpret the solution.
For this, do the following:
    1. Imports and definitions
    2. Generate problem matrices
    3. Formulate the LP
    4. Assemble the solution
    5. Plots and illustratons

More Information can be found e.g. on p. 134 in Scheduling by Michael L. Pinedo,
Springer Science & business Media (2016).
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
import scipy.linalg as scal
import matplotlib.pyplot as plt
import cvxpy as cp
import time



# ii) Definitions

n_machines=3
n_jobs=12
n_total=n_jobs*n_jobs*n_machines



"""
    2. Generate problem matrices ---------------------------------------------
"""


# i) Processing time matrix P

# Generate P by permuting the following vector. This is done simply for ease of
# generation
p_vec=np.array([[1,0.5,2,1,0.5,0.25,3,2,0.125,1,2,1]])   
P=np.bmat([[p_vec],[np.roll(p_vec,[0,-1])],[np.roll(p_vec,[0,-2])]])


# ii) Linear constraint matrix A

# A=np.ones([1,n_jobs*n_machines])
# for k in range(n_jobs-1):
#     A=scal.block_diag(A,np.ones([1,n_jobs*n_machines]))
A=np.ones([n_machines*n_jobs,1])
# A is designed such that A@np.flatten(x)=1
# A is designed such that x.T@A=1

# iii) Linear inequality matrix B

B=np.ones([1,n_jobs])
# B is designed such that B@(np.reshape(x,[n_machines*n_jobs,n_jobs]).T)<=1


# iv) Resource storage matrix

Q=np.zeros([n_machines,n_jobs,n_jobs])
for i in range(n_machines):
    for k in range(n_jobs):
        for j in range(n_jobs):
            Q[i,k,j]=(k+1)*P[i,j]
        
q_vec=np.reshape(Q,[1,n_total], order='F')



"""
    3. Formulate the LP ------------------------------------------------------
"""


# i) Define the decision variables

x=cp.Variable((n_machines*n_jobs,n_jobs),boolean=True)

objective=cp.Minimize(q_vec@cp.reshape(x,[n_total,1]))
cons=[]
# cons=[A@cp.reshape(x,[n_total,1])==np.ones([n_jobs,1])]
# cons=[A@cp.reshape(x,[n_total,1])<=np.ones([n_jobs,1])]
# cons=cons+[A@cp.reshape(x,[n_total,1])>=np.ones([n_jobs,1])]
cons=cons+[x.T@A>=np.ones([n_jobs,1])]
cons=cons+[x.T@A<=np.ones([n_jobs,1])]
cons=cons+[B@x.T<=np.ones([1,n_machines*n_jobs])]


# ii) Solve the Lp

t_0=time.time()
prob=cp.Problem(objective,cons)
prob.solve(verbose=True)
t_1=time.time()

Delta_time=t_1-t_0
x_opt=x.value



"""
    4. Assemble the solution -------------------------------------------------
"""


# i) Assemble schedule 

# The machine index changes rapidly, the time index more slowy.
# Extract the performed job sequence for each machine

jobs_machine_1=[]
jobs_machine_2=[]
jobs_machine_3=[]

for k in range(n_jobs):
    jobs_machine_1=jobs_machine_1+[np.where(x_opt[n_machines*k,:]==1)]
    jobs_machine_2=jobs_machine_2+[np.where(x_opt[n_machines*k+1,:]==1)]
    jobs_machine_3=jobs_machine_3+[np.where(x_opt[n_machines*k+2,:]==1)]


# ii) Objective value and constraints

obj_value_opt=q_vec@np.reshape(x_opt,[n_total,1],order='F')     # What is the objective value?
pri_res1=(x_opt.T@A-np.ones([n_jobs,1])==0)                     # Are the equalities satisfied?
pri_res2=x_opt@B.T<=np.ones([n_jobs*n_machines,1])              # Are the inequalities satisfied?
 


"""
    5. Plots and illustratons ------------------------------------------------
"""


# i) Plot x matrix

plt.figure(1,dpi=300)
plt.imshow(x_opt)
plt.title('The indicator matrix x')
plt.xlabel('job nr')
plt.ylabel('machine nr and timestep')

plt.figure(2, dpi=300)
plt.imshow(np.reshape(Q,[n_machines*n_jobs,n_jobs],order='F'))
plt.title(' Resource holding costs')
plt.xlabel('job_nr')
plt.ylabel('machine nr and timestep')

plt.figure(3, dpi=300)
plt.imshow(P)
plt.title(' Processing times')
plt.xlabel('job_nr')
plt.ylabel('machine nr')










