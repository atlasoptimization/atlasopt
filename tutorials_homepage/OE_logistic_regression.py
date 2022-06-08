"""
The goal of this script is to demonstrate logistic regression in a simple 
setting. We simulate failures of a mechanical system and model the failure 
probabilities and their evolution in time via logistic regression. 
For this, do the following:
    1. Imports and definitions
    2. Randomly generate data
    3. Logistic regression
    4. Plots and illustratons

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""



"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import numpy as np
from scipy.stats import bernoulli
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

np.random.seed(1)                            # Activate line for reproducibility 

# ii) Definitions - dimensions

n_sample=20             # nr of measurements
n_disc=50              # nr of points
n_total=n_disc**2

# iii) Definitions - auxiliary quantities

x=np.linspace(0,4,n_disc)
y=x

grid_x, grid_y=np.meshgrid(x,y)
ss=np.vstack((grid_x.flatten(), grid_y.flatten()))



"""
    2. Randomly generate data ------------------------------------------------
"""


# i) Measurement locations - 1D

sample_index=np.random.choice(np.linspace(0,n_disc-1,n_disc),[n_sample,1], replace=True).astype(int)
x_sample=x[sample_index]


# ii) Measurement locations - 2D

rand_int=np.random.choice(np.linspace(0,n_total-1, n_total).astype(int), size=[n_sample,1], replace=True)
index_tuples=np.unravel_index(rand_int, [n_disc,n_disc])


# iii) Create probability densities

lumbda_1=1
lumbda_2=0
failure_probability_1d=np.ones([1,n_disc])-np.exp(-lumbda_1*x)
failure_probability_2d=np.ones([n_disc,n_disc])-np.exp(-lumbda_1*grid_x-lumbda_2*grid_y)


# iii) Simulate the data 

data_1d=np.zeros([2,n_sample])
for k in range(n_sample):
    data_1d[0,k]=x_sample[k]
    data_1d[1,k]=bernoulli.rvs(failure_probability_1d[0,sample_index[k]],size=1)
    
data_2d=np.zeros([3,n_sample])
for k in range(n_sample):
    data_2d[0,k]=x[index_tuples[0][k]]
    data_2d[1,k]=x[index_tuples[1][k]]
    data_2d[2,k]=bernoulli.rvs(failure_probability_2d[index_tuples[0][k],index_tuples[1][k]],size=1)
    
# In data[1,:] we have: 1 = failure and 0= ok



"""
    3. Logistic regression ---------------------------------------------------
"""


# i) Invoke and train the model 1d

data_x=np.reshape(data_1d[0,:],[n_sample,1])
data_y=np.reshape(data_1d[1,:],[n_sample])

logreg_1d = LogisticRegression()
logreg_1d.fit(data_x,data_y)

data_1d_fail=data_x[np.where(data_y)]
data_1d_ok=data_x[np.where(1-data_y)]


# ii) Invoke and train the model 2d

data_x_2d=data_2d[0:2,:].T
data_y_2d=data_2d[2,:].T

logreg_2d = LogisticRegression()
logreg_2d.fit(data_x_2d,data_y_2d)

data_2d_fail=data_x_2d[np.where(data_y_2d)]
data_2d_ok=data_x_2d[np.where(1-data_y_2d)]


# iii) use model to predict

prediction_1d=logreg_1d.predict(np.reshape(x,[n_disc,1]))
prediction_proba_1d=logreg_1d.predict_proba(np.reshape(x,[n_disc,1]))

prediction_2d=logreg_2d.predict(ss.T)
prediction_proba_2d=logreg_2d.predict_proba(ss.T)

prediction_2d_reshaped=np.reshape(prediction_2d,[n_disc,n_disc],order='F')
prediction_proba_2d_reshaped=np.reshape(prediction_proba_2d[:,1],[n_disc,n_disc],order='F')



"""
    4. Plots and illustratons ------------------------------------------------
"""


# i) Plot 1d logistc regression

plt.figure(1,dpi=300)
plt.plot(x,prediction_1d,color='k', label='prediction')
plt.plot(x,prediction_proba_1d[:,1],color='k', linestyle=':', label='probability')
plt.scatter(data_1d_fail[:], np.ones(data_1d_fail.shape), s=90, label='data failure',facecolors='none', edgecolors='k',linewidths=2)
plt.scatter(data_1d_ok[:], np.zeros(data_1d_ok.shape), s=90,label='data ok',color='k',linewidths=2)
plt.title('Logistic regression 1D')
plt.xlabel('time ')
plt.ylabel('failure')
plt.legend()


# ii) Plot 2d logistc regression

plt.figure(2,dpi=300)
plt.imshow(np.rot90(prediction_proba_2d_reshaped),extent=[0,4,0,4])
plt.scatter(data_2d_fail[:,0], data_2d_fail[:,1], s=90, label='data failure',facecolors='none', edgecolors='k',linewidths=2)
plt.scatter(data_2d_ok[:,0], data_2d_ok[:,1], s=90,label='data ok',color='k',linewidths=2)
plt.title('Logistic regression 2D probability')
plt.xlabel('x ')
plt.ylabel('y')
plt.legend()



# iii) Plot 2d logistc regression decision boundary

plt.figure(3,dpi=300)
plt.imshow(np.rot90(prediction_2d_reshaped),extent=[0,4,0,4])
plt.scatter(data_2d_fail[:,0], data_2d_fail[:,1], s=90, label='data failure',facecolors='none', edgecolors='k',linewidths=2)
plt.scatter(data_2d_ok[:,0], data_2d_ok[:,1], s=90,label='data ok',color='k',linewidths=2)
plt.title('Logistic regression 2D boundary')
plt.xlabel('x ')
plt.ylabel('y')
plt.legend()














