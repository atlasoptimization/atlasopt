"""
The goal of this script is to solve a mini experiment design problem that figures
out a measurement distribution for minimizing uncertainty of a line fit.
For this, do the following:
    1. Imports and definitions
    2. Set up matrices
    3. Illustrate

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

n=100
n1=np.linspace(1,10,n)
n2=n1



"""
    2. Set up matrices -------------------------------------------------------
"""


# i) Design matrix and covariance matrices

A=np.array([[1,0],[1,1]])
A_p=np.linalg.pinv(A)

Sigma_l=lambda n1,n2: np.diag(np.array([1/n1,1/n2]))
Sigma_theta=lambda n1,n2: A_p@Sigma_l(n1,n2)@A_p.T


# ii) calculate scores for different combinations of n1, n2

sigma_theta_1=np.zeros([n,n])
sigma_theta_2=np.zeros([n,n])

for k in range(n):
    for l in range(n):
        Sigma_mat=Sigma_theta(n1[k],n2[l])
        sigma_theta_1[k,l]=Sigma_mat[0,0]
        sigma_theta_2[k,l]=Sigma_mat[1,1]

sigma_total=sigma_theta_1+sigma_theta_2



"""
    3. Illustrate ------------------------------------------------------------
"""

# i) Create Grid for feasible set
        
        
# iii) Plot image

fig, ax = plt.subplots(1, 3, figsize=(10, 5),dpi=300)
ax[0].imshow(np.rot90(sigma_theta_1),extent=(1,10,1,10),vmin=0.0, vmax=1)
ax[0].set_title('Unsicherheit $\sigma_{\Theta 1}^2$')
ax[1].imshow(np.rot90(sigma_theta_2),extent=(1,10,1,10), vmin=0.0, vmax=1)
ax[1].set_title('Unsicherheit $\sigma_{\Theta 2}^2$')
ax[2].imshow(np.rot90(sigma_total),extent=(1,10,1,10), vmin=0.0, vmax=1)
ax[2].set_title('Gesamtunsicherheit $\sigma_{\Theta 2}^2+\sigma_{\Theta 2}^2$')








