from __future__ import annotations
from typing import List, Callable, Dict, Any, Tuple, Union, Optional
from pathlib import Path
from torch import Tensor
import torch, torch.nn as nn, torch.nn.functional as F
import torch
import gymnasium as gym
from datasets import Dataset as HFDataset
from argparse import ArgumentParser
import os, shutil, math, logging 
from lightning import Fabric
from lightning.fabric.utilities import AttributeDict
from wandb.integration.lightning.fabric import WandbLogger
from datasets import Dataset as HFDataset
from argparse import ArgumentParser
import numpy as np
from dataclasses import dataclass, field

from lactchain.models.lightning_agent import LightningA2C
from lactchain.environments.grid_world import GridEnvironment, VectorizedGridWorld
from lactchain.models.critic import ValueFunction, ValueFunctionConfig, LoraConfigSettings
from lactchain.models.actor import LactChain, ActorConfig, Strategy
from lactchain.utils import configure_logger, unfold_list_of_lists
from lactchain.environments.grid_world import make_env, process_environment_outputs

'''GRAB OBSERVATION SET --> TORCH MULTINOMIAL --> SAMPLE BATCH OF STATES 
--> PASS INTO LLM TWICE --> ENV.RESET()
--> IF ASYNC, SEND FULL BATCH IN ELSE SEND IN SEQUENTIALLY
'''

# def batch_sample_states(sampled_states:Tensor, infos:list[str], batch_size:int) -> Tensor:
#     '''
#     Takes in sampled_state_tensor 
    
#     Returns a batch of sampled_states
#     Outputs:
#     states: list[Dict[str, int]]
#     infos: list[str]
#     '''
#     rand_batch_indices=torch.randint(0, sampled_states.size(0), (batch_size,))
    
#     sampled_state_tensor=sampled_states[rand_batch_indices]
#     sampled_states=VectorizedGridWorld.create_states_from_sampled_states(sampled_state_tensor)
#     sampled_infos=[infos[info] for info in rand_batch_indices]

#     return sampled_states, sampled_infos

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TORCH_LOGS']="+dynamo"
os.environ['TORCHDYNAMO_VERBOSE']='1'
# default actor path
ACTOR_PATH='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'
ACTOR_MODEL_TYPE='llama-3'
CRITIC_PATH='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11'

@dataclass
class DPOData: 
    '''Simple Dataclass to Store Data for DPO Dataset'''
    prompts:list[str]=field(
        default_factory=list
    )
    chosen:list[str]=field(
        default_factory=list
    )
    rejected:list[str]=field(
        default_factory=list
    )

def argparse(): 
    parser=ArgumentParser()
    parser.add_argument(
        '--actor_path', 
        type=str, 
        default=ACTOR_PATH, 
        help='''Path to frozen causal model'''
        )
    parser.add_argument(
        '--actor_model_type', 
        type=str, 
        default=ACTOR_MODEL_TYPE, 
        help='''Path to frozen causal model'''
        )
    parser.add_argument(
        '--critic_path', 
        type=str, 
        default=CRITIC_PATH, 
        help='''Path to trainable embedding model'''
        )
    parser.add_argument(
        '--global_num_samples',
        type=int, 
        default=10000,
        help='Number of states to sample globally'
    )
    parser.add_argument(
        '--per_rank_batch_size', 
        type=int, 
        default=32, 
        help='batch size for sampling states + infos'
    )
    parser.add_argument(
        '--resume_from_checkpoint', 
        type=str, 
        default=None, 
        help='Critic Checkpoint to use'
    )
    parser.add_argument(
        '--logging_save_path', 
        type=str, 
        default='training.log', 
        help='The file to save the logging file to'
    )
    parser.add_argument(
        '--gamma',
        type=float, 
        default=0.99, 
        help='''Discount factor ratio for calculating returns for Monte Carlo A2C'''
    )
    parser.add_argument(
        '--global_buffer_size', 
        type=int, 
        default=100, 
        help='''The desired max length of the buffer after gathering from all local ranks'''
        )
    args=parser.parse_args()
    return args

def sample_states_infos(env:VectorizedGridWorld, 
                        num_samples:int, 
                        batch_size:int
                        ) -> Tuple[List[Dict[str, Any]], List[str]]:
    '''
    Uses torch.Categorical to sample random (x, y, orientation) coordinates from an env 
    Then we turn this information into list of {x, y, orientation} and list of repeating strings that comprise
    the base info for the environment. Returned lists are of len batch_size 
    
    Inputs: 
    ======
    env: gym.Env 
        Gym environment to sample from...this can be a vector environment 
    num_samples:int 
        The number of samples to sample from the environment 
    
    Output: 
    ======
    sampled_states: list[Dict[str, Any]]
        Stores a list of dictionaries that encode the {'x', 'y', 'orientation'} coordinates 
    sampled_infos: list[str]
        Stores a list of strings that are passed into the causal model 
    '''

    distro_coord_space=env.coordinate_space_distro
    sampled_coords=distro_coord_space.sample((num_samples, 2))
    distro_orientation_space=env.orientation_space_distro
    sampled_orientations=distro_orientation_space.sample((num_samples,))
    
    infos=[env.environment_info]*num_samples
    sampled_states=torch.cat([sampled_coords, sampled_orientations.unsqueeze(1)], dim=1)
    
    rand_batch_indices=torch.randint(0, sampled_states.size(0), (batch_size,))
    sampled_state_tensor=sampled_states[rand_batch_indices]
    sampled_states=env.create_states_from_sampled_states(sampled_state_tensor)
    sampled_infos=[infos[info] for info in rand_batch_indices]
    
    return sampled_states, sampled_infos

def main():
    args=argparse()
    logger = configure_logger('debug', args.logging_save_path)
    wandb_logger=WandbLogger(project="my-project", offline=True)
    # Initialize Fabric
    fabric = Fabric(loggers=wandb_logger, accelerator='cpu')
    rank = fabric.global_rank # rank on global devices 
    world_size = fabric.world_size # total num devices 
    device = fabric.device
    local_rank=fabric.local_rank # rank on local node 
    
    actor_config=ActorConfig()
    lora_config=LoraConfigSettings()
    critic_config=ValueFunctionConfig()
    
    vector_env = gym.vector.AsyncVectorEnv([make_env for _ in range(args.per_rank_batch_size)])
    
    dummy_env=VectorizedGridWorld()
    
    agent=LightningA2C(args.actor_path, args.actor_model_type, 
                       actor_config, lora_config, 
                       args.critic_path, critic_config, args.gamma)
    logger.info(f'Model Trainable Params:\n{agent.model_trainable_params}, Device:\n{fabric.accelerator}')
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("critic_checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            fabric.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            fabric.print(f"Resuming from checkpoint {path}")
            full_checkpoint = fabric.load(os.path.join(args.output_dir, path))
            episode = int(path.split("-")[1])
            agent.load_state_dict(full_checkpoint['model'])
            # optimizer.load_state_dict(full_checkpoint['optimizer'])
    
    # agent, optimizer = fabric.setup(agent, optimizer)
    agent = fabric.setup(agent)
    per_rank_buffer_size=int(args.global_buffer_size / world_size)
    
    logger.info(f'SAMPLING {per_rank_buffer_size} STATES FOR RANK')
    
    per_rank_buffer_size=int(args.global_buffer_size / world_size)
    buffer_size=0
    
    data=DPOData()
    
    # collecting next observations via batch
    while buffer_size<per_rank_buffer_size:
        
        sampled_states, local_infos=sample_states_infos(dummy_env, per_rank_buffer_size, args.per_rank_batch_size)
        obs, info = vector_env.reset()
        
        batch_mapped_actions_1, actions_1, contexts_1, drop_indices=agent.sample_actions(sampled_states, local_infos)
        next_obs_1, rewards_1, _, _, info_1 = vector_env.step(batch_mapped_actions_1)
        next_obs_1, info_1=process_environment_outputs(next_obs_1, info_1)
        
        obs, info = vector_env.reset()
        
        batch_mapped_actions_2, actions_2, contexts_2, drop_indices=agent.sample_actions(sampled_states, local_infos)
        next_obs_2, rewards_2, _, _, info_2 = vector_env.step(batch_mapped_actions_2)
        next_obs_2, info_2=process_environment_outputs(next_obs_2, info_2)
        
        inputs_1=agent.compile_and_tokenize(next_obs_1, info_1)
        advantages_1=agent.calculate_advantages(rewards_1, inputs_1)
        
        inputs_2=agent.compile_and_tokenize(next_obs_2, info_2)
        advantages_2=agent.calculate_advantages(rewards_2, inputs_2)
        
        advantages=torch.stack([advantages_1, advantages_2]).tranpose(0, 1)
        chosen_indices=advantages.argmax(dim=1)
        joint_states=np.array([actions_1, actions_2])
        chosen=joint_states[np.arange(len(chosen_indices)), chosen_indices]
        rejected=joint_states[np.arange(len(chosen_indices)), ~chosen_indices]
    
        data.prompts.append(agent.compile_prompts(obs, info))
        data.chosen.append(chosen.tolist())
        data.rejected.append(rejected.tolist())
        breakpoint()
        buffer_size=len(unfold_list_of_lists(chosen))

    fabric.barrier()
    DATA={
        'prompts':data.prompts, 
        'chosen':data.chosen, 
        'rejected':data.rejected
    }
    gathered_data=fabric.all_gather(DATA)
    
    
    
    # for episode in range(args.num_episodes):
        
    #     fabric.barrier() # sync per episode
    #     logger.info(f'Starting Episode {episode+1} on all ranks...') if rank==0 else None
    #     obs, info = vector_env.reset()
    #     obs, info=process_environment_outputs(obs, info)
    #     observations=[]
    #     rewards=[]
    #     infos=[]
    #     steps_kept=0
    #     buffer_size=0
        
    #     with torch.autograd.set_detect_anomaly(True):
    #         with torch.no_grad(): 
    #             per_rank_buffer_size=int(args.global_buffer_size / world_size)
    #             logger.info(f'Collecting Experience for Rank {rank}...')
    #             step=0
    #             # since we will all_gather to make buffer, desired buffer_size = world_size * per_rank_observation_counts
    #             while buffer_size<=per_rank_buffer_size: 
    #                 step+=1
    #                 try: 
    #                     batch_mapped_actions, actions, contexts, drop_indices=agent.sample_actions(obs, info)
    #                     # TODO: if action is dropped, then send in default actions but do not collect 
    #                     next_obs, reward, done, truncated, info = vector_env.step(batch_mapped_actions)
    #                     next_obs, info=process_environment_outputs(next_obs, info)

    #                     if drop_indices:
    #                         filtered_obs=[next_obs.pop(drop_idx) for drop_idx in drop_indices]
    #                         filtered_rewards=[reward.pop(drop_idx) for drop_idx in drop_indices]
    #                         filtered_info=[info.pop(drop_idx) for drop_idx in drop_indices]
    #                         observations.append(filtered_obs)
    #                         rewards.append(filtered_rewards)
    #                         infos.append(filtered_info)
    #                     else: 
    #                         observations.append(next_obs)
    #                         rewards.append(reward)
    #                         infos.append(info)
                        
    #                     obs = next_obs
    #                     logger.info(f'''Successfully parsed {args.per_rank_batch_size-len(drop_indices)} actions out of batch size {args.per_rank_batch_size} in step {step}, adding to replay buffer for rank {rank}...''')
    #                     steps_kept+=1
    #                 except Exception as e: 
    #                     logger.error(f'''Error when collecting experience from rank {rank}:\n{e}\nDropping Full Batch for Step {step}...''')
    #                     continue
                    
    #                 buffer_size = len(np.concatenate(rewards))
    #                 logger.info(f'''Replay buffer size: {buffer_size} at step {step} for rank {rank}''')  
                    
    #     logger.info(f'''FOR RANK {rank}\nTOTAL STEPS:{step}\nSTEPS KEPT:{steps_kept}\nGATHERING DATA FROM RANK {rank}...''')
    #     fabric.barrier()
    #     local_data={
    #         'rewards':[reward for reward in rewards], 
    #         'observations':[observation for observation in observations], 
    #         'infos':[info for info in infos]
    #         }
    #     gathered_data = fabric.all_gather(local_data)
    
if __name__=="__main__": 
    
    main()