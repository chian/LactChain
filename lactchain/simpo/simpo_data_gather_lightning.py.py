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





def main():
    logger = configure_logger()
    wandb_logger=WandbLogger(project="my-project", offline=True)
    # Initialize Fabric
    fabric = Fabric(loggers=wandb_logger, accelerator='cpu')
    rank = fabric.global_rank # rank on global devices 
    world_size = fabric.world_size # total num devices 
    device = fabric.device
    local_rank=fabric.local_rank # rank on local node 
    
    args=argparse()
    actor_config=ActorConfig()
    lora_config=LoraConfigSettings()
    critic_config=ValueFunctionConfig()
    
    vector_env = gym.vector.AsyncVectorEnv([make_env for _ in range(args.per_rank_batch_size)])
    
    logger.info(f'')
    agent=LightningA2C(args.actor_path, actor_config, lora_config, 
                       args.critic_path, critic_config, args.gamma)
    logger.info(f'Model Trainable Params:\n{agent.model_trainable_params}, Device:\n{fabric.accelerator}')
    optimizer = agent.configure_optimizers(args.learning_rate)
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        breakpoint()
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
            optimizer.load_state_dict(full_checkpoint['optimizer'])
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    agent, optimizer = fabric.setup(agent, optimizer)
    
    for episode in range(args.num_episodes):
        
        fabric.barrier() # sync per episode
        logger.info(f'Starting Episode {episode+1} on all ranks...') if rank==0 else None
        obs, info = vector_env.reset()
        obs, info=process_environment_outputs(obs, info)
        observations=[]
        rewards=[]
        infos=[]
        steps_kept=0
        buffer_size=0
        
        with torch.autograd.set_detect_anomaly(True):
            with torch.no_grad(): 
                per_rank_buffer_size=int(args.global_buffer_size / world_size)
                logger.info(f'Collecting Experience for Rank {rank}...')
                step=0
                # since we will all_gather to make buffer, desired buffer_size = world_size * per_rank_observation_counts
                while buffer_size<=per_rank_buffer_size: 
                    step+=1
                    try: 
                        batch_mapped_actions, actions, contexts, drop_indices=agent.sample_actions(obs, info)
                        # TODO: if action is dropped, then send in default actions but do not collect 
                        next_obs, reward, done, truncated, info = vector_env.step(batch_mapped_actions)
                        next_obs, info=process_environment_outputs(next_obs, info)

                        if drop_indices:
                            filtered_obs=[next_obs.pop(drop_idx) for drop_idx in drop_indices]
                            filtered_rewards=[reward.pop(drop_idx) for drop_idx in drop_indices]
                            filtered_info=[info.pop(drop_idx) for drop_idx in drop_indices]
                            observations.append(filtered_obs)
                            rewards.append(filtered_rewards)
                            infos.append(filtered_info)
                        else: 
                            observations.append(next_obs)
                            rewards.append(reward)
                            infos.append(info)
                        
                        obs = next_obs
                        logger.info(f'''Successfully parsed {args.per_rank_batch_size-len(drop_indices)} actions out of batch size {args.per_rank_batch_size} in step {step}, adding to replay buffer for rank {rank}...''')
                        steps_kept+=1
                    except Exception as e: 
                        logger.error(f'''Error when collecting experience from rank {rank}:\n{e}\nDropping Full Batch for Step {step}...''')
                        continue
                    
                    buffer_size = len(np.concatenate(rewards))
                    logger.info(f'''Replay buffer size: {buffer_size} at step {step} for rank {rank}''')  
                    
        logger.info(f'''FOR RANK {rank}\nTOTAL STEPS:{step}\nSTEPS KEPT:{steps_kept}\nGATHERING DATA FROM RANK {rank}...''')
        fabric.barrier()
        local_data={
            'rewards':[reward for reward in rewards], 
            'observations':[observation for observation in observations], 
            'infos':[info for info in infos]
            }
        gathered_data = fabric.all_gather(local_data)