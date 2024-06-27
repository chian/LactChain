import torch
import torch.distributed as dist
import os, sys
import torch.multiprocessing as mp
from typing import Optional, Union, Dict, Any, List, Callable
import sys, os
from lactchain.environments.grid_world import GridEnvironment
import gymnasium as gym
from transformers import AutoModel
'''
Helpful Cuda-Torch Commands: 
- torch.cuda.is_available() -> bool: 
    checks if cuda devices are available
- torch.cuda.current_device() -> int: 
    returns the current device your process is on 
- torch.cuda.device_count() -> int: 
    returns the total amount of gpus that are available to use on a node
'''

PATH="/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11"

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
    
    
def gather_states(rank: int, world_size: int, sub_size: int):
    '''1. make a list of tensor of ones on rank [0] --> scatter to ranks [4, 6, 8]'''
    sub_ranks_1 = [0, 1, 2, 3]
    sub_group_1 = dist.new_group(sub_ranks_1)

    sub_ranks_2 = [2, 3]
    sub_group_2 = dist.new_group(sub_ranks_2)

    if rank in sub_ranks_1:
        env = create_env(rank)
        state, info = env.reset()
        env_state={'state':str(state), 'info':info}
        output = [None]*len(sub_ranks_1)
        dist.all_gather_object(output, env_state, group=sub_group_2)
        
        dist.barrier()
        print(f'Rank: {rank}, Received State: {output}') if rank in sub_ranks_2 else None
    
def init_process(rank:int, world_size:int, port:int, fn:Callable, backend:str='nccl') -> None:
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost' # '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    '''Note: the dist.init_process_group creates distributed process group. each process calls this function so that 
    each process understands the group it belongs to, the number of ranks/processes, world size
    '''
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    fn(rank, world_size, 3) # your function that is called
    
def create_env(rank:int): 
    from lactchain.environments.grid_world import GridEnvironment
    import gymnasium as gym
    print(f'starting environment at rank {rank}')
    env=GridEnvironment()
    return env

def main(): 

    port=get_open_port()
    world_size=torch.cuda.device_count()
    num_processes=world_size
    processes=[]
    backend='nccl'
    mp.set_start_method('spawn')

    for rank in range(num_processes): 
        process=mp.Process(target=init_process, args=(rank, world_size, port, step_1, backend))
        process.start()
        processes.append(process)

    for process in processes: 
        process.join()
        
if __name__=="__main__": 
    
    main()