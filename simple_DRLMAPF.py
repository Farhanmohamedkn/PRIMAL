#!/usr/bin/env python
# coding: utf-8

# # Pathfinding via Reinforcement and Imitation Multi-Agent Learning (PRIMAL)
# 
# While training is taking place, statistics on agent performance are available from Tensorboard. To launch it use:
# 
# `tensorboard --logdir train_primal`

# In[3]:


#this should be the thing, right?
from __future__ import division

import gym
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from od_mstar3 import cpp_mstar
from od_mstar3.col_set_addition import OutOfTimeError,NoSolutionError
import threading
import time
import scipy.signal as signal
import os
import GroupLock
import multiprocessing

import mapf_gym as mapf_gym
import pickle
import imageio

from ACNet_direct_pytorch import ACNet




import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

# Initialize environment
num_agents = 5  # Set the number of agents
DIAG_MVMT = False  # Allow diagonal movements
ENVIRONMENT_SIZE = (10,20)  # Size of the grid environment
GRID_SIZE = 10
OBSTACLE_DENSITY = (0,.2)
FULL_HELP = False
LR=1e-5
RNN_SIZE = 512


gameEnv = mapf_gym.MAPFEnv(num_agents=num_agents, DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=ENVIRONMENT_SIZE,
                           observation_size=GRID_SIZE, PROB=OBSTACLE_DENSITY, FULL_HELP=FULL_HELP)
print(gameEnv)

# Set up the model, optimizer, and other parameters
a_size = 5 + int(DIAG_MVMT)*4 # Define action size
model = ACNet('global', a_size, None, True, GRID_SIZE, 'global')
optimizer = torch.optim.NAdam(model.parameters(), lr=LR)

# Training parameters
max_episodes = 1000
max_steps = 200

# Training loop
for episode in range(max_episodes):
    print("Episode numberrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr=",episode)
    current_agent=1
    while(current_agent<=num_agents):
        hxs = torch.zeros(num_agents, RNN_SIZE)
        cxs = torch.zeros(num_agents, RNN_SIZE)

        for step in range(max_steps):
            
            state_tensor = torch.randn(5, 4, 10, 10)  # Permute to match Conv2d input shape
            goal_tensor = torch.randn(5, 3)

            with torch.no_grad():
                policy, value, (hxs, cxs), blocking, on_goal = model(state_tensor, goal_tensor, hxs, cxs)
                

            a_dist = policy.numpy()


            validActions          = gameEnv._listNextValidActions(current_agent)
            s                     = gameEnv._observe(current_agent)
            
            # action = policy.multinomial(1).data.numpy()  # Sample action
            # print("action=",action)
            
            # next_state, reward, done, _ = gameEnv._step((action))

            valid_dist = np.array([a_dist[0,validActions]])
            valid_dist /= np.sum(valid_dist)
            action_idx = np.random.choice(validActions, p=valid_dist.ravel())
            a = action_idx
            

            state, reward, done, nextActions, on_goal, blocking, valid_action = gameEnv._step((current_agent, a),episode=episode)
        
            # Convert to tensor
            reward_tensor = torch.tensor(reward, requires_grad=False)
            
            done_tensor = torch.tensor(done, requires_grad=False)
            # print('reward_tensor',reward_tensor, 'done_tensor', done_tensor)
            # Compute loss
            value_loss = F.mse_loss(value.squeeze(), reward_tensor)
            # print('value_loss',value_loss)
            advantage = reward_tensor - value.detach().squeeze()
            policy_loss = -(torch.log(policy.gather(1, torch.tensor(a, requires_grad=False).view(-1, 1))) * advantage.detach()).mean()
            # blocking_loss = F.binary_cross_entropy(blocking.squeeze(), torch.tensor(done))
            # on_goal_loss = F.binary_cross_entropy(on_goal.squeeze(), torch.tensor(done))
            
            # loss = value_loss + policy_loss + blocking_loss + on_goal_loss
            loss = value_loss + policy_loss 
            # print('loss', loss)
            # Perform backpropagation
            optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            
            # state = next_state
            
            if done:
                break

        current_agent+=1

    # state = gameEnv.reset()
    # print("state.reset",state.info)
   

    print(f"Episode {episode} finished with total reward {np.sum(reward)}")

print("Training complete")



# %%
