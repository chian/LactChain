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

'''FOR NOW, WE DO NOT DO PARALLELIZE. JUST GET A BASIC VERSION WORKING'''

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
    
    infos=[env.info]*BATCH_SIZE
    
    sampled_states=torch.cat([sampled_coords, sampled_orientations.unsqueeze(1)], dim=1)
    
    return sampled_states, infos


def sample_one_shot_trajectories(actor:MyLactChain=None, 
                                 critic:ValueFunction=None, 
                                 env:gym.Env=None, 
                                 sampled_states:Tensor=None, 
                                 infos:list[str]=None
                                 ):
    import random
    NUM_TRAJECTORIES=100
    
    rand_indices=torch.randint(0, sampled_states.size(0), (2,))
    sampled_state_tensor=sampled_states[rand_indices]

    prompts=[]
    chosens=[]
    rejects=[]
    for (x, y, orientation), info in zip(sampled_state_tensor, infos): 
        state={'x':x, 'y':y, 'orientation':orientation}
        actions, contexts=actor.sample_actions([state]*2,[info]*2)
        out1, reward1, done1, info1=env.step(actions[0])
        _, _ = env.reset()
        out2, reward2, done2, info2=env.step(actions[1])
        values=critic([out1, out2], [info1, info2])
        advantages=torch.tensor([reward1, reward2]) - values
        advantage_idx=torch.where(advantages>advantages.min().item())[0].item()

        dataset=HFDataset.from_dict({'prompt':[actor.compile_prompt(state, info)], 'chosen':[str(actions[advantage_idx])],'rejected':[str(actions[advantage_idx])]})
        breakpoint()

    
    # actions, contexts=actor.sample_actions(sampled_states, sampled_infos)
    # out1, reward1, done1, info1=env.step(actions[0])
    # env.reset()
    # out2, reward2, done2, info2=env.step(actions[1])
    # values=critic([out1, out2], [info1, info2])
    # advantages=torch.tensor([reward1, reward2]) - values
    # dataset=HFDataset.from_dict({'prompt':..., 'chosen':...,'rejected':...})
    
    breakpoint()
    ...
    




if __name__=="__main__": 
    
    LACTCHAIN_PATH='./models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'
    CRITIC_PATH="./models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11"
    
    env=GridEnvironment()
    actor_config=PolicyConfig()
    lora_config=LoraConfigSettings()
    actor=MyLactChain(LACTCHAIN_PATH, actor_config, lora_config)
    # actor=None
    
    critic_config=ValueFunctionConfig()
    critic=ValueFunction(CRITIC_PATH, critic_config)
    
    sampled_states, infos=sample_data(env=env)
    
    out=sample_one_shot_trajectories(actor=actor, critic=critic, sampled_states=sampled_states, infos=infos, env=env)
    
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

    