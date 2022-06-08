"""
The goal of this script is to train a TD3 RL algorithm. An object is falling with
constant speed and needs to be sttered to a certain point on the ground. The
dynamics is given by the usual differential equations governing the relation-
ship between acceleration, velocity, and position. The acceleration can be chosen
by the RL agent as reaction to a state observation.
For this, do the following
    1. Definitions and imports
    2. Initialize the environment class
    3. Step method
    4. Reset method
    5. Render method
    6. Train with stable baselines
    7. Summarize and plot results
    

Control landing environment:

This environment consists of a model of a falling object that can act by 
accelerating to the left or the right. Each step it falls closer to the ground 
and is punished by its deviation on the x axis from a target value. The parameters
passed to it upon its creation are the the action dimension and the target x value.

INPUTS
    Name                 Interpretation                             Type
dims            Array of form [n_action]                        Array
x_target        Target coordinate                               Number
action          The action to be performed during step          Index
 
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
from collections import namedtuple

from stable_baselines3.common.env_checker import check_env


# ii) Definitions

n_learn=30000                           # Determines the amount of learning - change
                                        # to adjust training time
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



"""
    2. Initialize the environment class ---------------------------------------
"""


class Control_landing_env(gym.Env):
    
    def __init__(self, dims, x_target):
        super(Control_landing_env, self).__init__()
        
        # i) Definition of fixed quantities
        
        z_max=10
        n_a=dims[0]
        x=1*(np.random.normal(0,1))
        vx=0*(np.random.normal(0,1))
        self.n_state=3
        
        # ii) Space definitions
        
        self.action_space = spaces.Box(low=-0.2, high=0.2,
                                        shape=(n_a,), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-10, high=10,
                                        shape=(self.n_state,), dtype=np.float32)
        
        
        # iii) Initialization of updeatable logging parameters
        
        self.state=np.hstack((z_max,x,vx))
        
        self.epoch=0
        self.max_epoch=z_max
        self.state_sequence=self.state
        self.action_sequence=[]
        self.reward_sequence=[]
        self.n_action=n_a
        self.dim=np.array([3,n_a])
        self.z_max=z_max
        self.x_target=x_target
        
        
        
    """
        3. Step method -------------------------------------------------------
    """
    
    
    def step(self,action):    
        
        
        # i) Dynamics and differential relations
        
        acceleration_value=action
        
        state_change=np.array([-1,self.state[2].item(),acceleration_value.item()])       
        state=self.state       
        new_state=state+state_change                   
        
        # ii) Logging of changes
        
        self.state=new_state
        self.state_sequence=np.vstack((self.state_sequence,new_state))
        self.action_sequence.append(acceleration_value)
        
        
        # iii) Reward calculation and update
        
        reward=-np.abs(self.state[1]-self.x_target)
        self.reward_sequence=np.hstack((self.reward_sequence,reward))
        self.last_transition=Transition(state,action,new_state,reward)

        
        self.epoch=self.epoch+1
        if self.epoch==self.max_epoch:
            done=True
        else:
            done=False
        
        info = {}
        
        return self.state, reward, done, info
        
    
    
    """
        4. Reset method ------------------------------------------------------
    """
        
    def reset(self):
        
        
        # i) Generate starting position randomly
        
        x=1*(np.random.normal(0,1))
        vx=0*(np.random.normal(0,1))
        
        
        # ii) Reset updateable logging parameters
        
        self.state=np.hstack((self.max_epoch,x,vx))
        
        self.epoch=0
        self.state_sequence=self.state
        self.action_sequence=[]
        self.reward_sequence=[]
        
        observation=self.state
        return observation
        
        
    """
        5. Render method -----------------------------------------------------
    """
    
    
    # i) Render plot with trajectories
    
    def render(self, reward, mode='console'):
        
        plt.figure(1,dpi=300,figsize=(2,4))
        plt.plot(self.state_sequence[:,1],self.state_sequence[:,0])
        plt.title('Steered trajectory')
        plt.xlabel('x axis')
        plt.ylabel('z axis')
        print('Reward is ', reward) 
        
        
    # ii) Close method
    
    def close (self):
      pass

      


"""
    6. Train with stable baselines -------------------------------------------
"""


# i) Generate Environment
      
new_env=Control_landing_env(np.array([1]),0)
check_env(new_env)


# ii) Import and use stable baselines

from stable_baselines3 import PPO

model = PPO("MlpPolicy", new_env,verbose=1)
model.learn(total_timesteps=n_learn)



"""
    7. Summarize and plot results --------------------------------------------
"""


# i) Let the agent run on a few episodes and plot the results

obs = new_env.reset()

n_episodes=30

for k in range(n_episodes):
    done=False
    obs = new_env.reset()
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = new_env.step(action)

        if done:
            new_env.render(reward)
            # time.sleep(0.5)
            break












