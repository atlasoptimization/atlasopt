"""
The goal of this script is to solve a mini MDP illustrating a production plan
subject to stochastic state transitions
For this, do the following:
    1. Definitions and imports
    2. Formulate the markov decision process
    3. Assemble the solution
    4. Illustrate

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""



"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import numpy as np
import mdptoolbox
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ii) Definitions

n_i=10
n_d=10
n_s=n_i*n_d

n_a=5

inventory=np.linspace(0,n_i-1,n_i).astype(int)
demand=np.linspace(0,n_d-1,n_d).astype(int)
actions=np.linspace(0,n_a-1,n_a).astype(int)

sell_price=2
storage_price=1



"""
    2. Formulate the markov decision process ----------------------------------
"""


# i) Auxiliary functions

def transform_to_tuple(index):
    state_vec=np.unravel_index(index, [n_i,n_d])
    return state_vec



# ii) Define transitions

P=np.zeros([n_a,n_s,n_s])

for a in range(n_a):
    for k in range(n_s):
        for l in range(n_s):
            state=transform_to_tuple(k)
            state_prime=transform_to_tuple(l)
            
            # now create: 
            #               supply change = actual change since inventory cannot be <0, >n_i-1
            #               for this, use difference in orders and demands = theoretical change,
            #               supply shortage and oversupply exceeding storage
            
            order_demand_diff=a-state[1]
            supply_shortage=-np.min([state[0]+order_demand_diff,0])
            oversupply=np.max([state[0]+order_demand_diff-n_i+1,0])
            supply_change=order_demand_diff+supply_shortage-oversupply
        
            # now use: demand changes by at max 1 between timesteps
            #          new supply=oldsupply+nonnegative,bounded supplychange
            
            if np.abs(state[1]-state_prime[1])<=1 and state_prime[0]==state[0]+supply_change:
                P[a,k,l]=1

        P[a,k,:]=P[a,k,:]/np.sum(P[a,k,:])




# iii) Define rewards

R=np.zeros([n_s,n_a])

for k in range(n_s):
    for a in range(n_a):
            state=transform_to_tuple(k)
            
            order_demand_diff=a-state[1]
            supply_shortage=-np.min([state[0]+order_demand_diff,0])
            sold_units=state[1]-supply_shortage
            oversupply=np.max([state[0]+order_demand_diff-n_i+1,0])
            supply_change=order_demand_diff+supply_shortage-oversupply
            stored_units=state[0]+supply_change
            
            R[k,a]=np.max([0,sell_price*sold_units])-storage_price*stored_units  # Simple model
            # R[k,a]=np.max([0,sell_price*sold_units])+0.6*stored_units**2-5*stored_units  # for complex behavior


"""
    3. Assemble the solution -------------------------------------------------
"""


# i) Solve problem

Inventory_MDP=mdptoolbox.mdp.ValueIteration(P, R, 0.99)
Inventory_MDP.run()


# ii) Extract solution

Policy_tuple=Inventory_MDP.policy
Policy_mat=np.zeros([n_i,n_d])

for k in range(n_i):
    for l in range(n_d):
        index=np.ravel_multi_index((k,l), [n_i,n_d])       
        Policy_mat[k,l]=Policy_tuple[index]



"""
    4. Illustrate  -----------------------------------------------------------
"""


# i) Plot Policy matrix

plt.figure(1,dpi=300,figsize=(5,5))
ax = plt.subplot()
im = ax.imshow(Policy_mat)
plt.title('Policy matrix')
plt.ylabel('Inventory')
plt.xlabel('Demand')
divider = make_axes_locatable(ax)
cax = divider.append_axes("left", size="5%", pad=1)
cbar=plt.colorbar(im, cax=cax)
cbar.set_label('Action=Decision to buy')

plt.show()


# ii) Plot Reward matrix

plt.figure(2,dpi=300,figsize=(10,10))
ax = plt.subplot()
im = ax.imshow(R,interpolation='nearest', aspect='auto')
plt.title('Reward matrix')
plt.ylabel('Inventory and Demand')
plt.xlabel('Action')
divider = make_axes_locatable(ax)
cax = divider.append_axes("left", size="5%", pad=1)
cbar=plt.colorbar(im, cax=cax)
cbar.set_label('Reward')

plt.show()


# iii) Plot Transition matrices

plt.figure(3,dpi=300,figsize=(5,5))
ax = plt.subplot()
im = ax.imshow(P[0,:,:])
plt.title('Transition matrix for action 1')
plt.ylabel('Inventory and Demand')
plt.xlabel('Inventory and Demand')
divider = make_axes_locatable(ax)
cax = divider.append_axes("left", size="5%", pad=1)
cbar=plt.colorbar(im, cax=cax)
cbar.set_label('Transition probabilities')

plt.show()



plt.figure(4,dpi=300,figsize=(5,5))
ax = plt.subplot()
im = ax.imshow(P[2,:,:])
plt.title('Transition matrix for action 3')
plt.ylabel('Inventory and Demand')
plt.xlabel('Inventory and Demand')
divider = make_axes_locatable(ax)
cax = divider.append_axes("left", size="5%", pad=1)
cbar=plt.colorbar(im, cax=cax)
cbar.set_label('Transition probabilities')

plt.show()



