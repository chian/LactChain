import torch
import torch.distributed as dist
import os, sys
import torch.multiprocessing as mp
from typing import Optional, Union, Dict, Any, List
import sys, os
sys.path.append(os.getcwd()+'/../../../')
os.environ['HF_HOME']=os.getcwd()+'/../../../../'
from use_cases.mine.lactchain.config import BaseConfig
from classes.lactchain import LactChain, Context, Component
from use_cases.mine.lactchain.environment import GridEnvironment
from use_cases.mine.lactchain.critic import ValueFunction, ValueFunctionConfig, LoraConfig, LoraConfigSettings
from use_cases.mine.lactchain.lactchains import MyLactChain, PolicyConfig
    
    
def dist_train(world_size:int, rank:int, backend:str): 
    
    setup(world_size, rank, backend) # setup processes 
    
    print(f'TEST DIST TRAINING WITH WORLD SIZE: {world_size}, AND RANK: {rank}')
    ...
    
    
def dist_train_critic(world_size:int, rank:int, backend:str): 
    
    data_ranks=[0, 1, 2, 3]
    dist_data_group=dist.new_group(data_ranks)
    
    model_ranks=[4, 5, 6, 7]
    dist_model_group=dist.new_group(model_ranks)
    
    if rank in data_ranks: 
        env=GridEnvironment()
        obs, info = env.reset()
        
        
        ...
    
    ...
    

def train_critic(actor:MyLactChain, 
                 critic:ValueFunction, 
                 critic_optimizer: torch.optim.AdamW, 
                 critic_scheduler:torch.optim.lr_scheduler.CosineAnnealingLR, 
                 env:GridEnvironment, 
                 ): 
    
    TOTAL_PARAMS=sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f'STARTING TRAINING..., with model size {TOTAL_PARAMS}')
    
    MAX_EPISODES=10000
    GAMMA=0.99
    MAX_STEPS=100
    
    for episode in range(MAX_EPISODES): 
        
        rewards=[]
        values=[]
        
        state, info = env.reset()
        done=0
        steps=0
        while not done and steps<=MAX_STEPS:         
     
            print(f'state:{state}')
            action, context=actor.sample_action(state, info)
            
            value=critic(state, info)
            next_state, reward, done, info = env.step(action)
        
            rewards.append(reward)
            values.append(value)
            
            state=next_state
            steps+=1
    
    
    
    
    
    
def main(): 

    world_size=torch.cuda.device_count()
    num_processes=world_size
    processes=[]
    backend='nccl'
    mp.set_start_method('spawn')

    for rank in range(num_processes): 
        process=mp.Process(target=dist_train, args=(world_size, rank, backend))
        process.start()
        processes.append(process)

    for process in processes: 
        process.join()
        
        
if __name__=="__main__": 
    
    import gymnasium as gym
    # main()
    
    env=GridEnvironment()
    
    vect_env=gym.vector.AsyncVectorEnv(
        [lambda: GridEnvironment()]*3
    )
    
    obs, info = vect_env.reset()
    
    breakpoint()

