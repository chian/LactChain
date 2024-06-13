import torch
import torch.distributed as dist
import os, sys
import torch.multiprocessing as mp
from typing import Optional, Union, Dict, Any, List
import sys, os

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