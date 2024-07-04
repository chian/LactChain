from typing import List, Callable, Dict, Any, Tuple, Union, Optional
from pathlib import Path
from torch import Tensor
import torch, torch.nn as nn, torch.nn.functional as F
import torch
import gymnasium as gym
from datasets import Dataset as HFDataset
from argparse import ArgumentParser
import os, uuid
import lightning as pl
from lightning import Fabric
from torch.utils.data import DistributedSampler, RandomSampler, BatchSampler, DataLoader
from argparse import ArgumentParser
import itertools
from operator import itemgetter 
import numpy as np
from lactchain.models.lightning_agent import LightningA2C
from lactchain.environments.grid_world import GridEnvironment
from lactchain.models.critic import ValueFunction, ValueFunctionConfig
from lactchain.models.actor import LactChain, ActorConfig, Strategy, LoraConfigSettings
from lactchain.environments.grid_world import VectorizedGridWorld, make_env, process_environment_outputs

PathLike=Union[str, Path]

ACTOR_PATH='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'
CRITIC_PATH='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11'

def calc_returns_list(rewards:List[Tensor]) -> List[Tensor]:
    '''Takes a list of returns in trajectory and computes the return R_t for t in trajectory
    Input: Sequence[int] -> Output: Sequence[torch(int)]
    '''
    gamma=0.99
    returns=[]
    R = 0
    for r in rewards[::-1]:
        R = (r + gamma*R).clone()
        returns.insert(0, R)
    return returns

def unfold_list_of_lists(list_of_list:list[list[Any]]) -> list: 
    unfolded_list=list(itertools.chain.from_iterable(list_of_list))
    return unfolded_list

def train(
    fabric: Fabric,
    agent: LightningA2C,
    optimizer: torch.optim.Optimizer,
    data: Dict[str, Tensor],
    args: ArgumentParser,
):        
    rewards=torch.stack(unfold_list_of_lists(data['rewards']))
    observations=unfold_list_of_lists(data['observations'])
    infos=unfold_list_of_lists(data['infos'])
    breakpoint()
    indexes=list(range(len(data['rewards'])))
    sampler = DistributedSampler(
        indexes, num_replicas=fabric.world_size, rank=fabric.global_rank, shuffle=True
    )
    # sampler = RandomSampler(indexes)
    sampler = BatchSampler(sampler, batch_size=args.per_rank_batch_size, drop_last=False)
    
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(args.update_epochs):
            sampler.sampler.set_epoch(epoch)
            for batch_indices in sampler:
                optimizer.zero_grad()
                selected_rewards=rewards[[batch_indices]]
                selected_obs=itemgetter(*batch_indices)(observations)
                selected_infos=itemgetter(*batch_indices)(infos)
                batch={'rewards':selected_rewards, 'observations':selected_obs, 'infos':selected_infos}
                loss = agent.training_step(batch)                
                print(f"Batch {batch_indices}: loss = {loss.item()}")
                fabric.backward(loss)
                optimizer.step()
        
def argparse(): 
    args=ArgumentParser()
    args.add_argument('--num_envs', type=int, default=4)
    args.add_argument('--actor_path', type=str, default=ACTOR_PATH)
    args.add_argument('--critic_path', type=str, default=CRITIC_PATH)
    args.add_argument('--gamma', type=float, default=0.99)
    args.add_argument('--learning_rate', type=float, default=1e-4)
    args.add_argument('--num_steps', type=int, default=4)
    args.add_argument('--per_rank_batch_size', type=int, default=2)
    args.add_argument('--update_epochs', type=int, default=10)
    args.add_argument('--max_grad_norm', type=float, default=1.0)
    return args.parse_args()

def main():
    # Initialize Fabric
    fabric = Fabric()
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    
    args=argparse()
    # Environment setup
    vector_env = gym.vector.AsyncVectorEnv([make_env for _ in range(args.num_envs)])
    
    actor_config=ActorConfig()
    lora_config=LoraConfigSettings()
    critic_config=ValueFunctionConfig()
    
    agent=LightningA2C(args.actor_path, actor_config, lora_config, 
                       args.critic_path, critic_config, args.gamma)
    
    optimizer = agent.configure_optimizers(args.learning_rate)
    agent, optimizer = fabric.setup(agent, optimizer)
    
    obs, info = vector_env.reset()
    obs, info=process_environment_outputs(obs, info)
    observations=[]
    rewards=[]
    actions=[]
    values=[]
    infos=[]
    with torch.autograd.set_detect_anomaly(True):
        with torch.no_grad(): 
            for step in range(args.num_steps):
                try: 
                    mapped_actions, actions, contexts=agent.sample_actions(obs, info)
                    next_obs, reward, done, truncated, info = vector_env.step(mapped_actions)
                    next_obs, info=process_environment_outputs(next_obs, info)
                    # value=agent.calculate_value(next_obs, info)
                    # print(value)
                    
                    observations.append(next_obs)
                    rewards.append(reward)
                    # actions.append(mapped_actions)
                    infos.append(info)
                    # values.append(value)
                    obs = next_obs
                except Exception as e: 
                    print(f'Lightning Agent Error {e} dropping step {step}...')
                    pass
        
    fabric.barrier()
    local_data={
        'rewards':[reward for reward in rewards], 
        # 'values':torch.cat(values).to(fabric.device)
        'observations':[observation for observation in observations], 
        'infos':[info for info in infos]
        }
    
    gathered_data = fabric.all_gather(local_data)
    print(f'mapped actions: {gathered_data}')
    train(fabric, agent, optimizer, gathered_data, args)    
    

if __name__=="__main__": 
    
    main()