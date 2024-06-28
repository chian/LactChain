import gymnasium as gym
import torch, os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.multiprocessing import Queue
import torch.multiprocessing as mp
from typing import Callable, Any

from lactchain.distributed.dist_utils import create_env, get_open_port, cleanup
from lactchain.distributed.dist_critic_dataset import ExperienceBuffer
from lactchain.models.critic import ValueFunction, ValueFunctionConfig
from lactchain.models.actor import LactChain, ActorConfig, LoraConfigSettings


def gather_sampled_states(rank: int, 
                          world_size: int, 
                          queue:Queue, 
                          buffer:ExperienceBuffer,
                          ):
    '''Collect env state, info from env.reset() in ranks [0, 1, 2, 3]
    All_gather --> [2, 3] 
    '''
    sub_ranks_1 = list(range(world_size))
    sub_group_1 = dist.new_group(sub_ranks_1)

    num_steps=20
    done=False
    batch_size=1000
    env = create_env(rank)
    
    actor = None
    critic = None

    if rank == 0:
        device=torch.cuda.current_device()
        ACTOR_PATH='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'
        actor_config=ActorConfig()
        lora_config=LoraConfigSettings()
        actor = LactChain(ACTOR_PATH, actor_config, lora_config).to(device)
    elif rank == 1:
        device=torch.cuda.current_device()
        CRITIC_PATH="/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11"
        critic_config=ValueFunctionConfig()
        critic = ValueFunction(CRITIC_PATH, critic_config).to(device)
    
    dist.barrier()
    if rank in sub_ranks_1:
        while not done and len(buffer)<=batch_size: 
            state, info = env.reset()
            environment_output={'state':state, 'info':info}
            
            # Gather environment_output from all ranks on rank 0
            if rank == 0:
                gathered_outputs = [None] * world_size
            else:
                gathered_outputs = None
                
            dist.gather_object(environment_output, gather_list=gathered_outputs, dst=0, group=sub_group_1)
            
            # output = [None]*len(sub_ranks_1)
            # dist.all_gather_object(output, environment_output, group=sub_group_1)
            
            if rank == 0:
                print(f'Output Data: {environment_output}')
            #     # Process the collected data using the model
            #     processed_data = []
            #     for data in output:
            #         state = data['state']
            #         info = data['info']
                    
            #         next_state, reward, done, info = actor.sample_actions()
            #         processed_data.append({'state': next_state, 'info': info})
                    
            #     # Place processed data on the queue
            #     for i in range(world_size):
            #         queue.put(processed_data[i])
            #     print(f'Data placed on rank {rank}: {processed_data}')
            # else:
            #     processed_data = [None] * world_size
            # # Broadcast the processed data from rank 0 to all ranks
            # dist.broadcast_object_list(processed_data, src=0, group=sub_group_1)
            
            # # Process the received data
            # received_data = processed_data[rank]
            # print(f'Rank: {rank}, Received Data: {received_data}')
            
            # buffer.add(received_data)  # Add received data to buffer
            
            dist.barrier()
            
    return buffer

def init_process(rank:int, world_size:int, port:int, backend:str, fn:Callable, 
                 queue, buffer
                 ) -> None:
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost' # '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    '''Note: the dist.init_process_group creates distributed process group. each process calls this function so that 
    each process understands the group it belongs to, the number of ranks/processes, world size
    '''
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    fn(rank, world_size, queue, buffer) # your function that is called


def main(): 
    port=get_open_port()
    world_size=torch.cuda.device_count()
    num_processes=world_size
    processes=[]
    backend='nccl'
    queue=Queue()
    buffer=ExperienceBuffer()
    
    try: 
        for rank in range(num_processes): 
            process=mp.Process(target=init_process, 
                               args=(rank, world_size, port, backend,
                                     gather_sampled_states,
                                     queue, buffer))
            
            process.start()
            processes.append(process)
            
        for process in processes: 
            process.join()
    except: 
        cleanup()


if __name__=="__main__": 
    
    mp.set_start_method('spawn')
    
    main()
        



