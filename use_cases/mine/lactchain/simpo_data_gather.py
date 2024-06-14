import sys, os
from typing import List, Callable, Dict, Any, Tuple
from torch import Tensor
import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym
from datasets import Dataset as HFDataset, concatenate_datasets
from argparse import ArgumentParser
sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain/')
from classes.lactchain import LactChain, Context, Component
from use_cases.mine.lactchain.environment import GridEnvironment
from use_cases.mine.lactchain.critic import ValueFunction, ValueFunctionConfig, LoraConfigSettings
from use_cases.mine.lactchain.lactchains import MyLactChain, PolicyConfig
from use_cases.mine.lactchain.dataset import DPODataset

'''FOR NOW, WE DO NOT DO PARALLELIZE. JUST GET A BASIC VERSION WORKING'''

def sample_data(env:gym.Env, num_samples:int) -> Tuple[Tensor, list[str]]:
    '''NOTE: GRAB OBSERVATION SET --> TORCH MULTINOMIAL --> SAMPLE 
    BATCH OF STATES --> PASS INTO LLM TWICE --> ENV.RESET()
    --> IF ASYNC, SEND FULL BATCH IN ELSE SEND IN SEQUENTIALLY 
    '''    
    print(f'SAMPLING STATE-ACTIONS...')
    coord_space_prob=torch.from_numpy(env.coord_set_probability)
    distro_coord_space=torch.distributions.Categorical(coord_space_prob)
    sampled_coords=distro_coord_space.sample((num_samples, 2))
    
    orientation_space_prob=torch.from_numpy(env.orientation_set_probability)
    distro_orientation_space=torch.distributions.Categorical(orientation_space_prob)
    sampled_orientations=distro_orientation_space.sample((num_samples,))
    infos=[env.info]*num_samples
    sampled_states=torch.cat([sampled_coords, sampled_orientations.unsqueeze(1)], dim=1)
    return sampled_states, infos


def sample_one_shot_trajectories(actor:MyLactChain=None, 
                                 critic:ValueFunction=None, 
                                 env:gym.Env=None, 
                                 sampled_states:Tensor=None, 
                                 infos:list[str]=None
                                 ):
    print(f'GENERATING DPO DATASET...')
    rand_indices=torch.randint(0, sampled_states.size(0), (2,))
    sampled_state_tensor=sampled_states[rand_indices]

    prompts=[]
    chosens=[]
    rejects=[]
    for (x, y, orientation), info in zip(sampled_state_tensor, infos): 
        state={'x':x, 'y':y, 'orientation':orientation}
        actions, _=actor.sample_actions([state]*2,[info]*2)
        out1, reward1, _, info1=env.step(actions[0])
        _, _ = env.reset()
        out2, reward2, _, info2=env.step(actions[1])
        values=critic([out1, out2], [info1, info2])
        advantages=torch.tensor([reward1, reward2]) - values
        breakpoint()
        advantage_idx=torch.where(advantages>advantages.min().item())[0].item()
        
        prompts.append(actor.compile_prompt(state, info))
        chosens.append(str(actions[advantage_idx]))
        rejects.append(str(actions[advantage_idx]))
    
    return {'prompt':prompts, 'chosen':chosens,'rejected':rejects}

def argparse() -> Any: 
    argparse=ArgumentParser()
    argparse.add_argument('--data_save_path', type=str, default='./')
    argparse.add_argument('--actor_path', type=str, 
                          default='./models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de')
    argparse.add_argument('--critic_path', type=str, 
                          default='./models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11')
    argparse.add_argument('--num_samples', type=int, default=100)
    args = argparse.parse_args()
    return args 

def main(): 
    
    args=argparse()
    env=GridEnvironment()
    actor_config=PolicyConfig()
    critic_config=ValueFunctionConfig()
    lora_config=LoraConfigSettings()

    actor=MyLactChain(args.actor_path, actor_config, lora_config)
    critic=ValueFunction(args.critic_path, critic_config)
    
    sampled_states, infos=sample_data(env=env, num_samples=args.num_samples)
    data=sample_one_shot_trajectories(actor=actor, critic=critic, env=env,
                                      sampled_states=sampled_states, infos=infos)
    
    if args.data_save_path: 
        dataset=HFDataset.from_dict(data)
        dataset.save_to_disk(args.data_save_path)
        print(f'Data saved at {args.data_save_path}')
    
    

if __name__=="__main__": 
    
    main()



    