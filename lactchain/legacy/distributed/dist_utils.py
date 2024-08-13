import torch
import torch.distributed as dist
# from torch.multiprocessing import Manager
import os, sys
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from typing import Optional, Union, Dict, Any, List, Callable
import sys, os
from lactchain.environments.grid_world import GridEnvironment
import gymnasium as gym
from transformers import AutoModel
from lactchain.models.actor import ActorConfig, LactChain, LoraConfigSettings
'''
Helpful Cuda-Torch Commands: 
- torch.cuda.is_available() -> bool: 
    checks if cuda devices are available
- torch.cuda.current_device() -> int: 
    returns the current device your process is on 
- torch.cuda.device_count() -> int: 
    returns the total amount of gpus that are available to use on a node
'''

# PATH="/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11"
ACTOR_PATH="/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de"

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
    
def create_env(rank:int): 
    from lactchain.environments.grid_world import GridEnvironment
    import gymnasium as gym
    print(f'starting environment at rank {rank}')
    env=GridEnvironment()
    return env
        
        
def actor_inference(rank: int, world_size: int, shared_data:list[Any], lock:Any): 
    sub_ranks_1 = [0, 1, 2, 3]
    sub_group_1 = dist.new_group(sub_ranks_1)

    sub_ranks_2 = [2, 3]
    sub_group_2 = dist.new_group(sub_ranks_2)
    if rank==2: 
        with lock:
            print(f'Rank: {rank}, Received State: {shared_data}')
            states=[data['state'] for data in shared_data]
            infos=[data['info'] for data in shared_data]
        
        config=ActorConfig()
        lora_config=LoraConfigSettings()
        device=torch.cuda.current_device()
        actor=LactChain(ACTOR_PATH, config, lora_config).to(device)
        print(f'Model device {device}')
        actions, contexts=actor.sample_actions(states, infos)
        print(f'Rank: {rank}, action: {actions}')
        
        output = [{'action':action, 'context':context} for action, context in zip(actions, contexts)]
        print(f'Length of actor output:{len(output)}')
        
        dist.broadcast_object_list(object_list=output, 
                                    src=rank, 
                                    group=sub_group_1
                                    )
        
        # shared_list=[element for element in output]
        with lock:  # Synchronize access to the shared data
            for i in range(len(sub_group_1)):
                shared_data[i] = output[i]
        print(f'Shared Data: {shared_data}')
        
    dist.barrier()
    

# def init_process(rank:int, world_size:int, port:int, backend:str, fn:Callable, 
#                  queue, buffer, 
#                  actor, critic, 
#                  ) -> None:
#     """ Initialize the distributed environment. """
#     os.environ['MASTER_ADDR'] = 'localhost' # '127.0.0.1'
#     os.environ['MASTER_PORT'] = str(port)
#     torch.cuda.set_device(rank % torch.cuda.device_count())
#     '''Note: the dist.init_process_group creates distributed process group. each process calls this function so that 
#     each process understands the group it belongs to, the number of ranks/processes, world size
#     '''
#     dist.init_process_group(backend, rank=rank, world_size=world_size)

#     fn(rank, world_size, queue, buffer, actor, critic) # your function that is called


def main(): 

    port=get_open_port()
    world_size=torch.cuda.device_count()
    num_processes=world_size
    processes=[]
    backend='nccl'
    mp.set_start_method('spawn')
    # shared_data = [None]  # Shared list to store data between functions
    # manager = mp.Manager()
    # shared_data = manager.list([0]*num_processes)  # Shared list to store data between functions
    # lock = manager.Lock()
    q = Queue()
    
    try: 
        for rank in range(num_processes): 
            process=mp.Process(target=init_process, args=(rank, world_size, port, 
                                                          gather_reset_states, 
                                                          q)
                               )
            process.start()
            processes.append(process)
        for process in processes: 
            process.join()
    except: 
        cleanup()

    # try: 
    #     for rank in range(num_processes): 
    #         process=mp.Process(target=init_process, args=(rank, world_size, port, 
    #                                                     gather_reset_states, shared_data, lock, backend))
    #         process.start()
    #         processes.append(process)
    #     for process in processes: 
    #         process.join()
    # except: 
    #     cleanup()
        
    # try: 
    #     for rank in range(num_processes): 
    #         process=mp.Process(target=init_process, args=(rank, world_size, port, 
    #                                                     actor_inference, shared_data, lock, backend))
    #         process.start()
    #         processes.append(process)
    #     for process in processes: 
    #         process.join()
    # except: 
    #     cleanup()
        
    
        
if __name__=="__main__": 
    
    main()