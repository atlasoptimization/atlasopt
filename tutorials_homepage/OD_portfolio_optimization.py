"""
The goal of this script is to demonstrate optimal choice of a set of products x
(stocks, insurances, commodities) whose price action is uncertain. The price 
action p is only known with respect to its expected value mu and covariance 
matrix Sigma; we also assume the distribution to be Gaussian. The goal is to 
maximize the expected revenue under the constraint that the risk of uneconomical 
outcomes is bounded by the probability beta. Overall it solves

max  Expected (p^Tx)
s.t. Probability(p^Tx <a)<beta
     x>=0 , 1^Tx=x_max
     
translated to the SDP

min  -mu^Tx
s.t. [ (mu - a)I                        Phi^{-1}(beta) Sigma^(1/2)x] >=0
     [ Phi^{-1}(beta) (Sigma^(1/2)x)^T          (mu - a)           ]
     
     1^Tx=x_max     x>=0
      
in the optimization variables x. Phi^{-1} is the inverse standard normal cdf 
and x_max the maximum amount of money spent on all of the products x.
This problem illustrates maximizing the profit of a production of products x
while bounding the probability of a critical income of 10k units from above
by 0.1%. Expected value and variations of the price action have been observed.

For this, do the following:
    1. Imports and definitions
    2. Generate matrices
    3. Formulate the semidefinite problem
    4. Assemble the solution
    5. Plots and illustratons

The specific SDP based formulation is taken from p 171 Convex optimization
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
from scipy.stats import norm
import cvxpy as cp
import matplotlib.pyplot as plt


# ii) Definitions

n_dim_x=10              # nr of random variables in vector x
a=10                    # revenue below which enterprise is not economical
beta=0.001              # equals 0.1 % probability of this revenue or less occuring
x_max=10                 # maximum amount of ressources available for production



"""
    2. Generate matrices -----------------------------------------------------
"""


# i) Generate moment data

# Increasing expected return but increasing risk
mu=np.linspace(1,5,n_dim_x)
Sigma=np.diag(np.linspace(1,5,n_dim_x))


# ii) Constraints

phi_inv_1=norm.ppf(beta)
beta_risky=0.1
phi_inv_2=norm.ppf(beta_risky)

U,S,V=np.linalg.svd(Sigma)
Sigma_sqrt=U@np.diag(np.sqrt(S))@U.T



"""
    3. Formulate the semidefinite problem ------------------------------------
"""


# i) Define variables

x_1=cp.Variable((n_dim_x,1),nonneg=True)
x_2=cp.Variable((n_dim_x,1),nonneg=True)

# ii) Define constraints

# Original semidefiniteness constraint
cons_1= [cp.bmat([[(mu.T@x_1-a)*np.eye(n_dim_x),phi_inv_1*Sigma_sqrt@x_1],[phi_inv_1*(Sigma_sqrt@x_1).T, cp.reshape((mu.T@x_1-a),[1,1])]])>>0]        
cons_2= [cp.bmat([[(mu.T@x_2-a)*np.eye(n_dim_x),phi_inv_2*Sigma_sqrt@x_2],[phi_inv_2*(Sigma_sqrt@x_2).T, cp.reshape((mu.T@x_2-a),[1,1])]])>>0]             

# capacity constraint
cons_1= cons_1 + [np.ones([n_dim_x]).T@x_1==x_max]
cons_2= cons_1 + [np.ones([n_dim_x]).T@x_2==x_max]


# iii) Define objective function

objective_1=cp.Minimize(-mu.T@x_1)
objective_2=cp.Minimize(-mu.T@x_2)



"""
    4. Assemble the solution -------------------------------------------------
"""


# i) Solve problem 1

prob_1=cp.Problem(objective_1,cons_1)
prob_1.solve(verbose=True)
x_opt_1=x_1.value


# ii) Solve roblem 2

prob_2=cp.Problem(objective_2,cons_2)
prob_2.solve(verbose=True)
x_opt_2=x_2.value


"""
    5. Plots and illustratons ------------------------------------------------
"""


# i) Report results

print('For beta=0.1%, the best portfolio is ', x_opt_1)
print('Its revenue achieves an expected value of' , -prob_1.value , 'and a stadard deviation of ', np.sqrt(x_opt_1.T@Sigma@x_opt_1).item())

print('For beta=10%, the best portfolio is ', x_opt_2)
print('Its revenue achieves an expected value of' , -prob_2.value , 'and a stadard deviation of ', np.sqrt(x_opt_2.T@Sigma@x_opt_2).item())



# # ii) Illustration of optimal solution


plt.figure(1,dpi=300)
plt.bar(np.linspace(1,n_dim_x,n_dim_x),x_opt_1.squeeze()/x_max, color='k')
plt.title(r'Optimal portfolio variables (conservative), $\beta$ = 1 %')
plt.xlabel('Index nr')
plt.ylabel('Proportion of portfolio')

plt.figure(2,dpi=300)
plt.bar(np.linspace(1,n_dim_x,n_dim_x),x_opt_2.squeeze()/x_max, color='k')
plt.title(r'Optimal portfolio variables (risky), $\beta$ = 10 %')
plt.xlabel('Index nr')
plt.ylabel('Proportion of portfolio')
































