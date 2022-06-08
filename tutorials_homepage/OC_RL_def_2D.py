"""
The goal of this script is to train a TD3 RL algorithm on the random deformation 
task and compare the cumulative rewards to the ones gathered by alternative 
discretization strategies.
For this, do the following
    1. Definitions and imports
    2. Train with stable baselines
    3. Apply alternative methods
    4. Summarize and plot results
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.
"""


"""
    1. Definitions and imports -----------------------------------------------
"""


# i) Import basics and custom environment

import numpy as np
import time
import matplotlib.pyplot as plt
import OC_RL_class_def_2D_env as def_2D


# ii) Import stable baselines

from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env


# iii) Initialize and check

# np.random.seed(2)                                  # Activate line for reproducibility 
n_learn=30000                           # Determines the amount of learning - change
                                        # to adjust training time
def_2D_env=def_2D.Env()
def_2D_env.reset()
check_env(def_2D_env)



"""
    2. Train with stable baselines -------------------------------------------
"""


# i) Train a TD3 Model

# # You can uncomment the following to train a new model. The default is to import
# # a pretrained one

# start_time=time.time()
# model = TD3("MlpPolicy", def_2D_env,verbose=1, seed=0)
# model.learn(total_timesteps=n_learn)
# end_time=time.time()

# model.save('./OC_RL_trained_benchmark_def_2D_new')

model=TD3.load('./OC_RL_trained_benchmark_def_2D')



"""
    3. Apply alternative methods ---------------------------------------------
"""


# Note: All actions are in [-1,1]x[-1,1] and get mapped to [0,1]x[0,1] by 
# the environment translating input actions from the symmetric box space 
# [-1,1]x[-1,1] to indices

# i) Grid based sampling

def grid_based_sampling(environment):
    grid_x1=np.kron(np.array([-1/3, 1/3, 1]),np.array([1, 1, 1]))
    grid_x2=np.kron(np.array([1, 1, 1]), np.array([-1/3, 1/3, 1]))
    grid=np.vstack((grid_x1, grid_x2))
    action=grid[:,environment.epoch]
    return action



"""
    4. Summarize and plot results --------------------------------------------
"""



# i) Summarize results in table

n_episodes_table=10
table=np.zeros([n_episodes_table,2])


# Grid based sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_2D_env.reset()
    while done ==False:
        action = grid_based_sampling(def_2D_env)
        obs, reward, done, info = def_2D_env.step(action)

        if done:
            table[k,0]=reward
            break


# RL sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_2D_env.reset()
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = def_2D_env.step(action)

        if done:
            table[k,1]=reward
            break


# ii) Illustrate results

n_episodes=1

for k in range(n_episodes):
    done=False
    obs = def_2D_env.reset()
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = def_2D_env.step(action)

        if done:
            def_2D_env.render(reward)
            # time.sleep(0.5)
            break

mean_summary=np.mean(table,axis=0)
std_summary=np.std(table,axis=0)

print(' Reward means of different methods')
print(mean_summary)
print(' Reward standard_deviations of different methods')
print(std_summary)
# print('Time for RL procedure = ', end_time-start_time ,'sec')


# iii) Compare to grid based sampling

x_grid=np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]])
x_grid_plot=np.array([[0,0],[0,0.5],[0,1],[0.5,0],[0.5,0.5],[0.5,1],[1,0],[1,0.5],[1,1]])

plt.figure(2,dpi=300)
plt.imshow(def_2D_env.fun.T, extent=[0,1,1,0])
plt.scatter(x_grid_plot[:,0],x_grid_plot[:,1])
plt.colorbar()
plt.title('Grid based spatial measurements')
plt.xlabel('x1 axis')
plt.ylabel('x2 axis')

n_max_epoch=def_2D_env.max_epoch
for k in range(n_max_epoch):    
    def_2D_env.step(x_grid[k,:])

print('Measured locations are', x_grid_plot)
print(' Measurements are', def_2D_env.f_measured[n_max_epoch])
