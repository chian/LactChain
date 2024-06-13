import sys, os
from typing import List, Callable, Dict, Any
from torch import Tensor
import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym
from datasets import Dataset as HFDataset, concatenate_datasets

sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain/')
from classes.lactchain import LactChain, Context, Component
from use_cases.mine.lactchain.environment import GridEnvironment
from use_cases.mine.lactchain.critic import ValueFunction, ValueFunctionConfig, LoraConfigSettings
from use_cases.mine.lactchain.lactchains import MyLactChain, PolicyConfig
from use_cases.mine.lactchain.dataset import DPODataset


def sample_data(actor:MyLactChain=None, 
                critic:ValueFunction=None, 
                env:gym.Env=None, 
                dataset:DPODataset=None, 
                gamma:float=None
                ):
    '''NOTE: GRAB OBSERVATION SET --> TORCH MULTINOMIAL --> SAMPLE 
    BATCH OF STATES --> PASS INTO LLM TWICE --> ENV.RESET()
    --> IF ASYNC, SEND FULL BATCH IN ELSE SEND IN SEQUENTIALLY 
    '''
    NUM_EPISODES=1000
    MAX_STEPS=15
    
    BATCH_SIZE=64
    
    coord_space_prob=torch.from_numpy(env.coord_set_probability)
    distro_coord_space=torch.distributions.Categorical(coord_space_prob)
    sampled_coords=distro_coord_space.sample((BATCH_SIZE, 2))
    
    orientation_space_prob=torch.from_numpy(env.orientation_set_probability)
    distro_orientation_space=torch.distributions.Categorical(orientation_space_prob)
    sampled_orientations=distro_orientation_space.sample((BATCH_SIZE,))
    
    sampled_states=torch.cat([sampled_coords, sampled_orientations.unsqueeze(1)], dim=1)
    
    breakpoint()




if __name__=="__main__": 
    
    env=GridEnvironment()
    
    sample_data(env=env)
    
    breakpoint()

    # LACTCHAIN_PATH='./models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'
    # CRITIC_PATH="./models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11"
    
    # lora_config=LoraConfigSettings()
    # lactchain_config=PolicyConfig()
    # lactchain=MyLactChain(LACTCHAIN_PATH, lactchain_config, lora_config, './')
    
    # critic_config=ValueFunctionConfig()
    # critic=ValueFunction(CRITIC_PATH, critic_config)
    
    # env=GridEnvironment()
    
    # dataset=None
    
    # sample_data(env=env)



    breakpoint()

    # obs, info = env.reset()

    # action, context=lactchain.sample_action(obs, info)
    # next_obs, reward, done, info = env.step(action)
    # breakpoint()

    # critic=ValueFunction(critic_config)
    # dataset=DPODataset()
    # breakpoint()

    # dataset=sample_experience(env, 5, lactchain, critic, dataset)

    breakpoint()

    