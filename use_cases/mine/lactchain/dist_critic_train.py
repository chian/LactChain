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

'''
Helpful Cuda-Torch Commands: 
- torch.cuda.is_available() -> bool: 
    checks if cuda devices are available
- torch.cuda.current_device() -> int: 
    returns the current device your process is on 
- torch.cuda.device_count() -> int: 
    returns the total amount of gpus that are available to use on a node
'''
def get_open_port() -> int:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("",0))
        s.listen(1)
        port = s.getsockname()[1]
        s.close()
        return port

def get_cuda_devices() -> list[int]: 
    '''returns list of devices'''
    num_devices=torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_list=[device for device in range(num_devices)]
    return device_list

def setup(world_size:int, rank:int, backend:str): 
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29600'

    dist.init_process_group(world_size=world_size, rank=rank, backend=backend) # needs to run for each rank 
    # this initialized distr. package perprocess, tells the process info on world size, what rank it is, and comm backend 

def cleanup(): 
    dist.destroy_process_group() # destroy process group per comm
    
    
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

