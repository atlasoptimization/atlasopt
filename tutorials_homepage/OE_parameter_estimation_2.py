"""
The goal of this script is to illustrate parameter estimation and solve fitting
trajectories and vector fields.
For this, do the following:
    1. Imports and definitions
    2. Simulate data
    3. Solve the optimal estimation problems
    4. Illustrate

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""



"""
    1. Imports and definitions ------------------------------------------------
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt
import copy

# np.random.seed(1)                      # Activate line for reproducibility 


# ii) Definitions

n_t=100
n_z=20
n_sample=20
n_sample_vfield=20
n_funs_traj=8
n_funs_vfield=4
n_funs_vfield_total=2*n_funs_vfield**2

t=np.linspace(-2,2,n_t)
z=np.linspace(-2,2,n_z)

zz1, zz2=np.meshgrid(z,z)



"""
    2. Simulate data ---------------------------------------------------------
"""


# i) Set up functions trajectory

x_true_traj=np.random.normal(0,1,[n_funs_traj,1])
g=[]

def make_fun_x(k):
    def f(t):
        return np.array([[t**(k)],[0]])
    return f

def make_fun_y(k):
    def f(t):
        return np.array([[0],[t**(k)]])
    return f

gx=[make_fun_x(k) for k in range(np.round(n_funs_traj/2).astype(int))]
gy=[make_fun_y(k) for k in range(np.round(n_funs_traj/2).astype(int))]
                
g=gx+gy
        
# ii) Generate full dataset trajectory

g_vec=np.zeros([n_t,n_funs_traj,2])

for k in range(n_t):
    for l in range(n_funs_traj):
        g_vec[k,l,:]=np.squeeze(g[l](t[k]))
    
traj_true=np.squeeze(x_true_traj.T@np.transpose(g_vec,[0,1,2]))
noise=np.random.normal(0,0.2,[n_t,2])

data_vec=traj_true+noise


# iii) Set up functions vector field

x_true_vfield=np.random.normal(0,1,[n_funs_vfield_total,1])
g_vfield=[]
gx_vfield=[]
gy_vfield=[]

def make_fun_x_vfield(k,l):
    def f(s,t):
        return np.array([[s**(k)*t**(l)],[0]])
    return f

def make_fun_y_vfield(k,l):
    def f(s,t):
        return np.array([[0],[s**(k)*t**(l)]])
    return f

for k in range(n_funs_vfield):
    for l in range(n_funs_vfield):
        gx_vfield.append(make_fun_x_vfield(k,l))
        gy_vfield.append(make_fun_y_vfield(k,l))
                
g_vfield=gx_vfield+gy_vfield


# iv) Generate full dataset vector field

g_vec_vfield=np.zeros([n_z, n_z, n_funs_vfield_total,2])

for k in range(n_z):
    for l in range(n_z):
        for m in range(n_funs_vfield_total):
            g_vec_vfield[k,l,m,:]=np.squeeze(g_vfield[m](z[k],z[l]))
    
vfield_true=np.squeeze(x_true_vfield.T@np.transpose(g_vec_vfield,[0,1,2,3]))
noise_vfield=np.random.normal(0,0.2,[n_z, n_z, 2])

data_vec_vfield=vfield_true+noise_vfield


# v) Sample data points

rand_ind=np.round(np.random.uniform(0,n_t-1,[n_sample,1])).astype(int)
t_data=np.squeeze(t[rand_ind])
l_data=np.squeeze(data_vec[rand_ind,:])

rand_ind=np.round(np.random.uniform(0,n_z-1,[n_sample_vfield,2])).astype(int)
z_data_vfield=np.hstack((np.reshape(zz1[rand_ind[:,0], rand_ind[:,1]],[n_sample_vfield,1]),np.reshape(zz2[rand_ind[:,0], rand_ind[:,1]],[n_sample_vfield,1])))
l_data_vfield=np.hstack((np.reshape(data_vec_vfield[rand_ind[:,0], rand_ind[:,1],0],[n_sample_vfield,1]),np.reshape(data_vec_vfield[rand_ind[:,0], rand_ind[:,1],1],[n_sample_vfield,1])))



"""
    3. Solve the optimal estimation problems ----------------------------------
"""


# i) Create matrices and vectors - trajectory estimation

G_traj=np.zeros([n_sample*2, n_funs_traj])
G_full=np.zeros([n_t*2, n_funs_traj])

for k in range(n_sample):
    for l in range(n_funs_traj):
        G_traj[2*k:2*k+2,l]=np.squeeze(g[l](t_data[k]))

for k in range(n_t):
    for l in range(n_funs_traj):
        G_full[2*k:2*k+2,l]=np.squeeze(g[l](t[k]))        
    
    
l_vec=l_data.reshape([2*n_sample,1])
    

# ii) Solve for optimal parameters -trajectory estimation

x_opt=np.linalg.pinv(G_traj)@l_vec
l_hat=np.reshape(G_full@x_opt,[n_t,2])



# iii) Create matrices and vectors - vectorfield estimation

G_vfield=np.zeros([n_sample_vfield*2, n_funs_vfield_total])
G_full_vfield=np.zeros([2*n_z**2, n_funs_vfield_total])

for k in range(n_sample_vfield):
    for l in range(n_funs_vfield_total):
        G_vfield[2*k:2*k+2,l]=np.squeeze(g_vfield[l](z_data_vfield[k,0], z_data_vfield[k,1]))

for k in range(n_z**2):
    for l in range(n_funs_vfield_total):
        ind1,ind2=np.unravel_index(k, [n_z,n_z])
        G_full_vfield[2*k:2*k+2,l]=np.squeeze(g_vfield[l](zz1[ind1,ind2], zz2[ind1,ind2]))       
    
    
l_vec_vfield=l_data_vfield.reshape([2*n_sample_vfield,1])
    

# iv) Solve for optimal parameters - vectorfield estimation

x_opt_vfield=np.linalg.pinv(G_vfield)@l_vec_vfield
l_hat_vfield=np.reshape(G_full_vfield@x_opt_vfield,[n_z**2,2])
l_hat_vfield_x=np.reshape(l_hat_vfield[:,0],[n_z,n_z])
l_hat_vfield_y=np.reshape(l_hat_vfield[:,1],[n_z,n_z])



"""
    4. Illustrate ------------------------------------------------------------
"""  


# i) Basic quantities

markersize=50


# ii) Trajectory figure

f1=plt.figure(1, dpi=500)
# plt.scatter(traj_true[:,0],traj_true[:,1])
plt.plot(traj_true[:,0],traj_true[:,1],color='k',label='True trajectory')
plt.scatter(l_data[:,0],l_data[:,1],color='k', marker='o', s=markersize, label='Observations')
# plt.scatter(data_vec[:,0],data_vec[:,1],color='k',marker='*', s=markersize)
plt.xlabel('x direction')
plt.ylabel('y direction')
f1.legend()
plt.title('True trajectory and observations')


f2=plt.figure(2,dpi=500)
plt.plot(g_vec[:,1,0], g_vec[:,1,1], color='k')
plt.plot(g_vec[:,5,0], g_vec[:,5,1], color='k')
plt.xlabel('x direction')
plt.ylabel('y direction')
plt.title('Basis functions')

f3=plt.figure(3,dpi=500)
plt.plot(g_vec[:,1,0], g_vec[:,6,1], color='k')
plt.plot(g_vec[:,2,0], g_vec[:,5,1], color='k')
plt.xlabel('x direction')
plt.ylabel('y direction')
plt.title('Basis functions')

f4=plt.figure(4,dpi=500)
plt.plot(g_vec[:,1,0], g_vec[:,7,1], color='k')
plt.plot(g_vec[:,3,0], g_vec[:,5,1], color='k')
plt.xlabel('x direction')
plt.ylabel('y direction')
plt.title('Basis functions')

f5=plt.figure(5,dpi=500)
plt.scatter(l_data[:,0],l_data[:,1],color='k', marker='o', s=markersize, label='Beobachtungen')
plt.plot(l_hat[:,0],l_hat[:,1], color='k',label='Geschaetzte Trajektorie')
plt.xlabel('x direction')
plt.ylabel('y direction')
f1.legend()
plt.title('Estimated trajectory')
        

# iii) Vector field figure

f6=plt.figure(6, dpi=500)
# plt.quiver(zz1,zz2, data_vec_vfield[:,:,0], data_vec_vfield[:,:,1])
plt.quiver(z_data_vfield[:,0], z_data_vfield[:,1], l_data_vfield[:,0], l_data_vfield[:,1])
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.title('Observations vector field')

f7=plt.figure(7, dpi=500)
plt.quiver(zz1,zz2, g_vec_vfield[:,:,1,0], g_vec_vfield[:,:,1,1])
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.title('Basis function')

f8=plt.figure(8, dpi=500)
plt.quiver(zz1,zz2, g_vec_vfield[:,:,10,0], g_vec_vfield[:,:,10,1])
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.title('Basis function')

f9=plt.figure(9, dpi=500)
plt.quiver(zz1,zz2, g_vec_vfield[:,:,3,0], g_vec_vfield[:,:,23,1])
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.title('Basis function')

f10=plt.figure(10, dpi=500)
plt.quiver(zz1, zz2, l_hat_vfield_x, l_hat_vfield_y, color='k')
plt.quiver(z_data_vfield[:,0], z_data_vfield[:,1], l_data_vfield[:,0], l_data_vfield[:,1], color='r')
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.title('Observations vector field')
























