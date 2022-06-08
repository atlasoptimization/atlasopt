"""
The goal of this script is to demonstrate optimal bounding of probabilities
based on known information regarding the moments of the underlying probability
distribution. We formulate the problem of finding the best upper bound for 
P(x in C) where C is the complement to an open polyhedron defined by a sequence
of linear inequalities. Overall it solves

max P(x in C)
s.t. P is a probability distribution with known moments
     C is a known polyhedral constraint region
     
translated to the SDP

min tr(S@P) + q^Tmu +r
s.t. [P   q] >+ 0
     [q^T r] 
     
     [P     q]  >=tau_k [0        ak/2]
     [q^T r-1]          [ak^T/2  -b_k]
   
     tau_k >=0 for all k=1, ... n_c
      
in the optimization variables P,q,r, tau. S and mu are the first and second moments.
This problem illustrates bounding the probability of a product constructed with 
an error rendering it unusable. The area of tolerance is modeled as a set of
constraints and the variations and covariations of the  individual properties 
have been observed.
 
For this, do the following:
    1. Imports and definitions
    2. Randomly generate matrices
    3. Formulate the semidefinite problem
    4. Assemble the solution
    5. Plots and illustratons

More Information can be found e.g. in pp.469 - 509 Handbook of semidefinite 
programming by H. Wolkowicz et. al., Springer Science & business Media (2003). 
The specific SDP based formulation is taken from pp 376 - 378 Convex optimization
by S. Boyd and L. Vandenberghe , Cambridge University Press (2004).
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
import matplotlib.pyplot as plt


# ii) Definitions

n_dim_x=10              # nr of random variables in vector x
n_cons=2*n_dim_x        # nr of constraints forming the polyhedron 

# For n_dim_x=1, one gets the classical Chebychev inequality 
# P(|x-mu|>k)<sigma^2/k^2 = 1/25 for our choices
# Note that higher dimensions increase the likelihood of x coming to lie outside
# of the polyhedron since there exist more possibilites for constraint violation.

"""
    2. Randomly generate matrices --------------------------------------------
"""


# i) Generate moment data

np.random.seed(1)

# Either randomly chosen empirical moments
B=np.random.normal(0,1,[n_dim_x,2*n_dim_x])
mu=np.mean(B,axis=1)
Sigma=B@B.T/n_dim_x

# Or fixed white noise moments
# mu=np.zeros([n_dim_x])
# Sigma=np.eye(n_dim_x)


# ii) Constraints

A_c=np.vstack((np.eye(n_dim_x),-np.eye(n_dim_x)))    # Constraint matrix, bounds polyhedron to [-5,5]
b_c=5*np.ones([n_cons])                                # Then A_c x <b is polyhedron bound



"""
    3. Formulate the semidefinite problem ------------------------------------
"""


# i) Define variables

P=cp.Variable((n_dim_x,n_dim_x),PSD=True)           # The Matrix variable
q=cp.Variable((n_dim_x))                            # The vector variable
r=cp.Variable((1),nonneg=True)                      # The scalar variable
tau=cp.Variable((n_cons),nonneg=True)               # The auxiliary variable


# ii) Define constraints

# Original semidefiniteness constraint
cons= [cp.bmat([[P,cp.reshape(q,[n_dim_x,1])],
                [cp.reshape(q,[1,n_dim_x]),cp.reshape(r,[1,1])]])>>0]                

for k in range(n_cons):
    LMI_mat_1=cp.bmat([[P,cp.reshape(q,[n_dim_x,1])],[cp.reshape(q,[1,n_dim_x]),cp.reshape(r-1,[1,1])]])
    LMI_mat_2=tau[k]*cp.bmat([[np.zeros([n_dim_x,n_dim_x]), 1/2*A_c[k,:].reshape([n_dim_x,1])],[1/2*A_c[k,:].reshape([1,n_dim_x]), -np.reshape(b_c[k],[1,1])]])    
    cons=cons+[LMI_mat_1>>LMI_mat_2]


# iii) Define objective function

objective=cp.Minimize(cp.trace(Sigma@P)+2*mu.T@q+r)


"""
    4. Assemble the solution --------------------------------------------------
"""


# i) Solve problem

prob=cp.Problem(objective,cons)
prob.solve(verbose=True)


# ii) Extract solution

P_opt=P.value
q_opt=q.value
r_opt=r.value
tau_opt=tau.value


"""
    5. Plots and illustratons -------------------------------------------------
"""


# i) Report probability bounds

print('The probability of lying outside of the polyhedron is at most ', prob.value)
print('The probability of lying inside of the polyhedron is at least' , 1-prob.value)


# ii) Illustration of optimal solution


plt.figure(1,dpi=300)
plt.imshow(P_opt)
plt.title('Optimal parameter values')
plt.xlabel('Index nr')
plt.ylabel('Index nr')


plt.figure(2,dpi=300)
plt.imshow(np.reshape(q_opt,[n_dim_x,1]))
plt.ylabel('Index nr')

plt.figure(3,dpi=300)
plt.imshow(np.reshape(q_opt,[1,n_dim_x]))
plt.xlabel('Index nr')


plt.figure(4,dpi=300)
plt.imshow(np.reshape(r_opt,[1,1]))





























