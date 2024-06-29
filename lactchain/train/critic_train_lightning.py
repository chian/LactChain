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
from torch.utils.data import DistributedSampler, RandomSampler, BatchSampler
from argparse import ArgumentParser

from lactchain.models.lightning_agent import LightningA2C
from lactchain.environments.grid_world import GridEnvironment
from lactchain.models.critic import ValueFunction, ValueFunctionConfig
from lactchain.models.actor import LactChain, ActorConfig, Strategy, LoraConfigSettings
from lactchain.environments.grid_world import VectorizedGridWorld, make_env, process_environment_outputs

PathLike=Union[str, Path]

ACTOR_PATH='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'
CRITIC_PATH='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11'

def train(
    fabric: Fabric,
    agent: LightningA2C,
    optimizer: torch.optim.Optimizer,
    data: Dict[str, Tensor],
    args: ArgumentParser,
):
    breakpoint()
    indexes = list(range(data["values"].shape[0]))
    if args.share_data:
        sampler = DistributedSampler(
            indexes, num_replicas=fabric.world_size, rank=fabric.global_rank, shuffle=True
        )
    else:
        sampler = RandomSampler(indexes)
    sampler = BatchSampler(sampler, batch_size=args.per_rank_batch_size, drop_last=False)
    breakpoint()
    for epoch in range(args.update_epochs):
        if args.share_data:
            sampler.sampler.set_epoch(epoch)
        for batch_idxes in sampler:
            loss = agent.training_step({k: v[batch_idxes] for k, v in data.items()})
            optimizer.zero_grad(set_to_none=True)
            fabric.backward(loss)
            fabric.clip_gradients(agent, optimizer, max_norm=args.max_grad_norm)
            optimizer.step()
        # agent.on_train_epoch_end()
        
def argparse(): 
    args=ArgumentParser()
    args.add_argument('--num_envs', type=int, default=4)
    args.add_argument('--actor_path', type=str, default=ACTOR_PATH)
    args.add_argument('--critic_path', type=str, default=CRITIC_PATH)
    args.add_argument('--gamma', type=float, default=0.99)
    args.add_argument('--learning_rate', type=float, default=1e-4)
    args.add_argument('--num_steps', type=int, default=5)
    args.add_argument('--per_rank_batch_size', type=int, default=4)
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
    for step in range(args.num_steps):
        try: 
            mapped_actions, actions, contexts=agent.sample_actions(obs, info)
            next_obs, reward, done, truncated, info = vector_env.step(mapped_actions)
            next_obs, info=process_environment_outputs(next_obs, info)
            value=agent.calculate_value(next_obs, info)
            print(value)
            
            observations.append(next_obs)
            rewards.append(reward)
            actions.append(mapped_actions)
            values.append(value)
            obs = next_obs
        except Exception as e: 
            print(f'Lightning Agent Error {e} dropping step {step}...')
            pass
    
    local_data={
            'rewards':torch.cat([torch.from_numpy(reward) for reward in rewards]).to(fabric.device), 
            'values':torch.cat(values).to(fabric.device)
            }
    
    gathered_data = fabric.all_gather(local_data)
    print(f'mapped actions: {gathered_data}')
    
    train(fabric, agent, optimizer, gathered_data, args)    
    

if __name__=="__main__": 
    
    main()

    # breakpoint()