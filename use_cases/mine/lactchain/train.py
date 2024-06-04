import sys, os
from typing import List, Callable, Dict, Any
from torch import Tensor
import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym

sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain/')
from classes.lactchain import LactChain, Context, Component
from use_cases.mine.lactchain.environment import GridEnvironment
from use_cases.mine.lactchain.critic import ValueFunction, ValueFunctionConfig, LoraConfig
from use_cases.mine.lactchain.lactchains import MyLactChain, PolicyConfig
from use_cases.mine.lactchain.dataset import DPODataset


def calc_returns(rewards:List[int], gamma:float) -> List[Tensor]:
    '''Takes a list of returns in trajectory and computes the return R_t for t in trajectory
    Input: Sequence[int] -> Output: Sequence[torch(int)]
    '''
    returns=[]
    R = 0
    for r in rewards[::-1]: 
        R = r + gamma*R
        returns.insert(0, torch.tensor(R))
    return returns

def simple_train(env:gym.Env, num_episodes:int, lactchain:MyLactChain, 
                 critic:ValueFunction, dataset:DPODataset): 
    '''Assumes one episode'''
    states=[]
    actions=[]
    next_states=[]
    rewards=[]

    done=False
    num_steps=1000
    steps=0
    obs, info=env.reset()
    while not done and steps<=num_steps: 

        action=lactchain.sample_action(obs, info)
    
    ...



def calculate_advantage(states:list[int], actions, rewards, next_states, critic, gamma) -> Dict[str, Any]: 
    states_tensor=torch.Tensor(states)
    next_states_tensor=torch.Tensor(next_states)
    rewards_tensor=torch.Tensor(rewards)
    values=critic(states)
    next_values=critic(next_states)
    advantages=(rewards_tensor + gamma*next_values - values)
    return ...

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
            breakpoint()
            next_obs, reward, done, info  = env.step(action)  # Execute the action

            states.append(str(obs)) # turn states into string
            actions.append(action)
            next_states.append(str(next_obs)) # turn states into string 
            rewards=rewards.append(reward)

            obs=next_obs
            steps += 1
            breakpoint()
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


if __name__=="__main__": 

    lactchain_config=PolicyConfig()
    critic_config=ValueFunctionConfig()

    lactchain=MyLactChain(lactchain_config, "mistralai/Mistral-7B-Instruct-v0.3", './')
    env=GridEnvironment()
    critic=ValueFunction(critic_config)
    dataset=DPODataset()
    breakpoint()

    dataset=sample_experience(env, 5, lactchain, critic, dataset)

    breakpoint()

    