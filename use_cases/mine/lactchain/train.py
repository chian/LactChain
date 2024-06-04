import sys, os
from typing import List, Callable
from torch import Tensor
import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym

sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain/')
from classes.lactchain import LactChain, Context, Component
from use_cases.mine.lactchain.environment import GridEnvironment
from use_cases.mine.lactchain.critic import ValueFunction
from use_cases.mine.lactchain.lactchains import MyLactChain
from use_cases.mine.lactchain.dataset import DPODataset

def calc_returns(rewards:List[int], cum_rewards:List[int], gamma:float) -> List[Tensor]:
    R = 0
    for r in rewards[::-1]: 
        R = r + gamma*R
        cum_rewards.insert(0, torch.tensor(R))
    return cum_rewards

def sample_experience(env:gym.Env, num_episodes:int, lactchain:MyLactChain, 
                      critic:ValueFunction, dataset:DPODataset): 
    states=[]
    actions=[]
    next_states=[]
    rewards=[]
    max_steps=100
    gamma=torch.Tensor(0.99,)

    for episode in range(num_episodes): 
        obs, info = env.reset()
        done = False
        steps = 0
        while not done and steps<=max_steps:

            action=lactchain.sample_actions(obs, info)
            next_obs, reward, done, info  = env.step(action)  # Execute the action

            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)
            rewards=rewards.append(reward)

            obs=next_obs
            steps += 1

        ##### advantage calculation
        states_tensor=torch.Tensor(states)
        next_states_tensor=torch.Tensor(next_states)
        rewards_tensor=torch.Tensor(rewards)

        values=critic(states)
        next_values=critic(next_states)
        advantages=(rewards_tensor + gamma*next_values - values)
        
    dataset.add_batch_transitions(states_tensor, actions, rewards_tensor, next_states_tensor, advantages)

    print(f'Episode completed in {steps} interactions')
    
    return dataset


def collect_experience(num_episodes:int, 
                       env:gym.Env, 
                       lactchain:LactChainA, 
                       critic:ValueFunction, 
                       memory:Memory, 
                       max_steps:int
                       ): 

    for episode in range(num_episodes): 
        done=False
        step=0
        obs = env.reset()

        while not done and step<=max_steps: 

            action = lactchain.propose_action(obs)
            next_obs, reward, done = env.step(action)
            memory.add_transition(obs, action, reward, next_obs)

            obs=next_obs
            step+=1



if __name__=="__main__": 



    breakpoint()
    