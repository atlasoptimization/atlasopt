"""
The goal of this script is to train a PPO RL algorithm. A random timeseries of
deformations is generated. At each point in time, The RL agent has to decide, 
if it wants to perform a measurement or not. As a consequence of a measurement,
some cost is incurred and the deformation at this timestep is revealed. At the 
end of the epoch, the reward is the sum of costs for all measurements + the 
root mean square error measurung unfaithfulness of the reconstruction. 
discretization strategies.
For this, do the following
    1. Definitions and imports
    2. Initialize the environment class
    3. Step method
    4. Reset method
    5. Render method
    6. Train with stable baselines
    7. Summarize and plot results
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.
"""


"""
    1. Definitions and imports -----------------------------------------------
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import spaces

from stable_baselines3.common.env_checker import check_env


# ii) Definitions

n_learn=100000                           # Determines the amount of learning - change
                                         # to adjust training time
meas_cost=0.001                          # Determines the cost of one measurement



"""
    2. Initialize the environment class ---------------------------------------
"""


class Measurement_Env(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, meas_cost):
    super(Measurement_Env, self).__init__()
    
    
    # i) Initialize fixed quantities
    
    t_max=50
    
    self.t_max=t_max
    self.n_action=2
    self.action_space = spaces.Discrete(self.n_action)
    self.observation_space = spaces.Box(low=np.array([0,-10,0,-10]), high=np.array([t_max,10,t_max,10]),
                                        shape=(4,), dtype=np.float32)
    self.max_epoch=t_max-1
    self.epoch=0
    self.t=np.linspace(0,1,t_max)
    self.meas_cost=meas_cost
    
    
    # ii) Generate data randomly
    
    K_mat=np.zeros([t_max,t_max])
    for k in range(t_max):
        for l in range(t_max):
            K_mat[k,l]=self.cov_fun(self.t[k],self.t[l])
    
    self.K_mat=K_mat
    self.fun=np.random.multivariate_normal(np.zeros([t_max]),K_mat)
    
    
    # iii) Initialize updateable logging parameters
    
    self.state=np.vstack((0,0,0,0)).flatten()
    self.state_sequence=self.state
    self.meas_sequence=np.empty([0,1])
    self.t_meas_sequence=np.empty([0,1])
    self.total_meas=0
    self.accum_reward=0
    self.fun_hat=0

  def cov_fun(self,t,s):
    # cov_val=np.exp(-np.abs((t-s)/1.2)**2)
    cov_val=np.exp(-np.abs((t-s)/0.2)**2)
    # cov_val=np.min(np.array([s,t]))
    return cov_val



    """
        3. Step method -------------------------------------------------------
    """
    

  def step(self, action):
   
      
   # i) Calculate state change and update logging parameters
   
    if action==0:
        meas_done=0
    else:
        meas_done=1
    self.total_meas=self.total_meas+meas_done
            
    self.epoch=self.epoch+1
    done = bool(self.epoch == self.max_epoch)
     
    if meas_done==0:
        state_change=np.array([+1,0,+1,0])
        new_state=self.state+state_change
    if meas_done==1:
        new_state=np.array([0,self.fun[self.epoch],self.state[0]+1,self.state[1]])
        self.meas_sequence=np.vstack((self.meas_sequence,np.array([new_state[1]])))  
        self.t_meas_sequence=np.vstack((self.t_meas_sequence,np.array([self.t[self.epoch]])))          
    
    self.state=new_state
    self.state_sequence=np.vstack((self.state_sequence,new_state))
    
    
    # ii) Evaluate root mean square error to quantify error of reconstruction
    
    rmse=0
    if done == True:
        
        K_t=np.zeros([self.t_max,self.total_meas])
        K_ij=np.zeros([self.total_meas,self.total_meas])
        
        for k in range(self.t_max):
            for l in range(self.total_meas):
                K_t[k,l]=self.cov_fun(self.t[k],self.t_meas_sequence[l])
                
        for k in range(self.total_meas):
            for l in range(self.total_meas):
                K_ij[k,l]=self.cov_fun(self.t_meas_sequence[k],self.t_meas_sequence[l])
        
        fun_hat=K_t@np.linalg.pinv(K_ij)@self.meas_sequence
        rmse=np.linalg.norm(self.fun-fun_hat.squeeze())
        self.fun_hat=fun_hat
    else:
        pass
    
    
    # iii) Calculate reward and pass it on
    
    reward= -meas_done*self.meas_cost-done*rmse
    self.accum_reward=self.accum_reward+reward

    # Optionally we can pass additional info, we are not using that for now
    info = {}

    return self.state.astype(np.float32), reward, done, info



    """
        4. Reset method ------------------------------------------------------
    """
    
  
  def reset(self):
      
      
      # i) Reset all the logging parameters to trivial and regenerate data

    t_max=self.t_max
    self.epoch=0
            
    self.fun=np.random.multivariate_normal(np.zeros([t_max]),self.K_mat)
    
    self.state=np.vstack((0,0,0,0)).flatten()
    self.state_sequence=self.state
    self.meas_sequence=np.empty([0,1])
    self.t_meas_sequence=np.empty([0,1])
    self.total_meas=0
    self.accum_reward=0
    self.fun_hat=0
    
    observation=self.state
    
    return observation.astype(np.float32)  # reward, done, info can't be included



    """
        5. Render method -----------------------------------------------------
    """
    
    
    # i) Plot estimated ad true deformation values

  def render(self, reward, mode='console'):
    if mode != 'console':
      raise NotImplementedError()
    plt.figure(1,dpi=300)
    plt.plot(self.t,self.fun,linestyle='solid',color='0',label='true function')
    plt.plot(self.t,self.fun_hat,linestyle='dashed',color='0',label='estimated function')
    plt.scatter(self.t_meas_sequence,self.meas_sequence,label='measurements')
    plt.title('Measurement times for cost = %1.5f ' % self.meas_cost)
    plt.xlabel('Time')
    plt.ylabel('Function value')
    # plt.legend()
    print(reward)
    
  def close (self):
      pass
      


"""
    6. Train with stable baselines -------------------------------------------
"""


# i) Generate Environment
      
meas_env=Measurement_Env(meas_cost)
check_env(meas_env)


# ii) Import and use stable baselines

from stable_baselines3 import PPO

model = PPO("MlpPolicy", meas_env,verbose=1)
model.learn(total_timesteps=n_learn)



"""
    7. Summarize and plot results --------------------------------------------
"""


# i) Let the agent run on a few episodes and plot the results

obs = meas_env.reset()

n_episodes=3

for k in range(n_episodes):
    done=False
    obs = meas_env.reset()
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = meas_env.step(action)

        if done:
            meas_env.render(meas_env.accum_reward)
            # time.sleep(0.5)
            break












