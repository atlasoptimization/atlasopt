"""
The goal of this script is to illustrate the behavior of the Newton method when
faced with multiple minima.
For this, do the following:
    1. Definitions and imports
    2. Iterate Newton methods
    3. Plots and illustrations

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""



"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt


# ii) Definitions

n=500
t_1=np.linspace(-2,2,n)                # Activate different t_1, t_2 extents for different zoom levels
t_2=np.linspace(-2,2,n)

# t_1=np.linspace(-2,0,n)
# t_2=np.linspace(-0.3,0.3,n)

# t_1=np.linspace(-2,-1.4,n)
# t_2=np.linspace(-0.3,0.3,n)

# t_1=np.linspace(-1.85,-1.75,n)
# t_2=np.linspace(-0.27,-0.17,n)

z_1=1
z_2=np.exp((2*np.pi*1j)/3)
z_3=np.exp((4*np.pi*1j)/3)



"""
    2. Iterate Newton method -------------------------------------------------
"""

# i) Loop over initial conditions

Newton_iterate=lambda z: z-(z**3-1)/(3*z**2)
Iteration_matrix=np.zeros([n,n])

for k in range(n):
    for l in range(n):
        z=t_1[k]+t_2[l]*1j

# ii) Perform Newton

        for m in range(10):
            z=Newton_iterate(z)
        
        if np.abs(z-z_1) < 0.01:
             Iteration_matrix[k,l]=1
        elif np.abs(z-z_2) < 0.01:
             Iteration_matrix[k,l]=2
        elif np.abs(z-z_3) < 0.01:
             Iteration_matrix[k,l]=3



"""
    3. Plots and illustrations -----------------------------------------------
"""

fig=plt.figure(dpi=500)
im=plt.imshow(Iteration_matrix.T, extent=[np.min(t_1),np.max(t_1),np.min(t_2),np.max(t_2)])
plt.xlabel('real axis')
plt.ylabel('imaginary axis')

        


