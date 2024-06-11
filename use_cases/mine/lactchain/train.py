import sys, os
from typing import List, Callable, Dict, Any
from torch import Tensor
import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym
from datasets import Dataset as HFDataset, concatenate_datasets

sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain/')
from classes.lactchain import LactChain, Context, Component
from use_cases.mine.lactchain.environment import GridEnvironment
from use_cases.mine.lactchain.critic import ValueFunction, ValueFunctionConfig, LoraConfig
from use_cases.mine.lactchain.lactchains import MyLactChain, PolicyConfig
from use_cases.mine.lactchain.dataset import DPODataset



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

def sample_experience(env:gym.Env, num_episodes:int, lactchain:MyLactChain): 
    
    trajectories=[]
    MAX_EPISODES=num_episodes
    GAMMA=0.99
    MAX_STEPS=5

    while len(trajectories)<=MAX_EPISODES:

        states=[]
        actions=[]
        next_states=[]
        rewards=[]
        contexts=[]
        
        state, info = env.reset()
        done=False
        step=0
        while not done and step<=MAX_STEPS: 

            action, context = lactchain.sample_action(state, info)
            next_state, reward, done, next_info = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            contexts.append(context)

            state=next_state
            info=next_info

            trajectory={'states':states,
                        'actions':actions,
                        'rewards':rewards, 
                        'next_states':next_states,
                        'contexts':contexts
                        }
            step+=1
            print(f'Doing Episode {len(trajectories)+1}, on Step {step}')

        trajectories.append(trajectory)

    return trajectories


if __name__=="__main__": 

    lactchain_config=PolicyConfig()
    critic_config=ValueFunctionConfig()

    lactchain=MyLactChain(lactchain_config, "mistralai/Mistral-7B-Instruct-v0.3", './')
    # critic=ValueFunction(critic_config)
    env=GridEnvironment()
    trajectories=sample_experience(env, 2, lactchain)
    datasets=[HFDataset.from_dict(trajectory) for trajectory in trajectories]
    master_dataset=concatenate_datasets(datasets)

    breakpoint()

    obs, info = env.reset()

    action, context=lactchain.sample_action(obs, info)
    next_obs, reward, done, info = env.step(action)
    breakpoint()

    critic=ValueFunction(critic_config)
    dataset=DPODataset()
    breakpoint()

    dataset=sample_experience(env, 5, lactchain, critic, dataset)

    breakpoint()

    