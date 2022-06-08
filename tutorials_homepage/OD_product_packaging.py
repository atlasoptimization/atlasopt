
"""
The goal of this script is to illustrate a mini product design problem that figures
out a length configuration for minimizing material usage of a package.
For this, do the following:
    1. Imports and definitions
    2. Calculate area usage
    3. Plots and illustrations

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.
"""



"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Import

import numpy as np
import matplotlib.pyplot as plt


# ii) Definitions

F=lambda x1, x2: 2*(x1*x2+1/x1+1/x2) 
s=np.linspace(0.25,2,100)
s1,s2=np.meshgrid(s,s)



"""
    2. Calculate area usage --------------------------------------------------
"""


# i)  Direct calculation

FF=np.zeros([100,100])
for k in range(100):
    for l in range(100):
        FF[k,l]=F(s1[k,l],s2[k,l])
        
        
        
"""
    3. Plots and illustrations -----------------------------------------------
"""  


# i) Area usage plot

plt.figure(1, dpi=300)
ax=plt.imshow(FF)
cbar=plt.colorbar(ax)
cbar.set_label('Area used [m2]')
plt.xlabel('Dimensions x1') 
plt.ylabel('Dimensions x2') 
plt.title('Area usage for packaging')   
        
        
        
        
        
        
        
        
        
        
        
        