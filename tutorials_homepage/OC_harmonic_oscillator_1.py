"""
The goal of this script is to illustrate the damped harmonic oscillators behavior
under different conditions on the damping coefficient.
For this, do the following:
    1. Definitions and imports
    2. Simulate sequentially
    3. Plots and illustrations

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""


"""
    1. Definitions and imports -----------------------------------------------
"""


# i) Imports

import numpy as np
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


# ii) Run a simulation

x_full=np.zeros([2,n])
x_full[0,0]=1

for k in range(1,n):
    x_full[:,k]=x_full[:,k-1]+Delta_t*A@x_full[:,k-1]
    
    

"""
    3. Plots and illustrations -----------------------------------------------
"""


# i) Figure on temporal behavior

plt.figure(1,dpi=300)
plt.plot(t,x_full[0,:],color='k')
plt.title('Temporal behavior for  damping coefficient c=%1.1f' %c)
plt.xlabel('Time t')
plt.ylabel('Position x')


# ii) Figure on phasespace behavior

plt.figure(2,dpi=300)
plt.plot(x_full[0,:],x_full[1,:],color='k')
plt.title('Temporal behavior for  damping coefficient c=%1.1f' %c)
plt.xlabel('Position x')
plt.ylabel('Velocity $\dot{x}$')






