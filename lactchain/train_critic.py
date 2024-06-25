from typing import List, Callable, Dict, Any, Tuple, Union, Optional
from pathlib import Path
from torch import Tensor
import torch, torch.nn as nn, torch.nn.functional as F
import torch
import gymnasium as gym
from datasets import Dataset as HFDataset
from argparse import ArgumentParser

from lactchain.environments.grid_world import GridEnvironment
from lactchain.models.critic import ValueFunction, ValueFunctionConfig, LoraConfigSettings
from lactchain.models.actor import LactChain, ActorConfig, Strategy


PathLike=Union[str, Path]

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

def train_critic(actor:LactChain, 
                 critic:ValueFunction, 
                 critic_optimizer: torch.optim.AdamW, 
                 critic_scheduler:torch.optim.lr_scheduler.CosineAnnealingLR, 
                 env:GridEnvironment, 
                 args:ArgumentParser,
                 ): 
    TOTAL_PARAMS=sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f'STARTING TRAINING..., with model size {TOTAL_PARAMS}')
    
    for episode in range(args.num_episodes): 
        
        rewards=[]
        values=[]
        state, info = env.reset()
        done=0
        steps=0
        
        while not done and steps<=args.max_steps:         
     
            action, context=actor.sample_action(state, info)
            value=critic(state, info)
            next_state, reward, done, info = env.step(action)
            
            print(f'STATE:{state}')
            print(f'ACTION: {action}')
        
            rewards.append(reward)
            values.append(value)
            
            state=next_state
            steps+=1
            
        cumulative_returns=calc_returns(rewards, args.gamma)
        cumulative_returns = torch.tensor(cumulative_returns)
        cumulative_returns = (cumulative_returns - cumulative_returns.mean()) /\
                                (cumulative_returns.std() + 1e-12)
        
        critic_losses=[]
        for R, value in zip(cumulative_returns, values): 
            critic_losses.append(F.smooth_l1_loss(R, value)) # advantage = R - value
            
        critic_loss=torch.stack(critic_losses).mean()
        
        critic_optimizer.zero_grad()
        critic_loss.backward() 
        critic_optimizer.step()
        
        if episode % 10 == 0: 
            critic_scheduler.step()
        
        print(f'Episode {episode+1} Loss: {critic_loss}')
        print(f'Episode {episode+1} Total Reward: {cumulative_returns.mean()}')
        
    if args.model_save_path: 
        actor.save_model(critic, args.model_save_path)
        
def argparse() -> Any:
    argparse=ArgumentParser()
    argparse.add_argument('--recycle_weights', type=bool, default=False)
    argparse.add_argument('--data_save_path', type=str, default='./datasets/dataset_2')
    argparse.add_argument('--actor_path', type=str,
                          default='./models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de')
    argparse.add_argument('--critic_path', type=str,
                          default='./models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11')
    argparse.add_argument('--num_samples', type=int, default=100)
    argparse.add_argument('--batch_size', type=int, default=16)
    args = argparse.parse_args()
    return args

def main():
    
    ACTOR_PATH='/nfs/lambda_stor_01/homes/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0417f4babd26db0b5ed07c1d0bc85658ab526ea3'
    CRITIC_PATH='/nfs/lambda_stor_01/homes/bhsu/2024_research/models/models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11'
    DEFAULT_ACTOR_PATH='./'
    DEFAULT_CRITIC_PATH='./'
    
    args=argparse()
    if args.recycle_weights: 
        args.actor_path=DEFAULT_ACTOR_PATH
        args.critic_path=DEFAULT_CRITIC_PATH
    else:
        args.actor_path=ACTOR_PATH
        args.critic_path=CRITIC_PATH
    print(args.actor_path)
    print(args.critic_path)
    
    lora_config=LoraConfigSettings()
    actor_config=ActorConfig()
    critic_config=ValueFunctionConfig()
    
    env=GridEnvironment()
    
    actor=LactChain(args.actor_path, actor_config, lora_config)
    
    critic=ValueFunction(args.critic_path, critic_config)
    
    critic_optimizer=torch.optim.AdamW(critic.parameters(), lr=1e-4)
    critic_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=10)

    train_critic(actor, critic, critic_optimizer, critic_scheduler, env, args)
    
    
if __name__=="__main__": 
    
    main()