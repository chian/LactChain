from typing import List, Callable, Dict, Any, Tuple, Union, Optional
from pathlib import Path
from torch import Tensor
import torch
import gymnasium as gym
from argparse import ArgumentParser
import lightning as pl
from lightning import Fabric
from torch.utils.data import DistributedSampler, BatchSampler
from argparse import ArgumentParser
import itertools
from operator import itemgetter 
import numpy as np
from pydantic import Field
from lightning.fabric.utilities import AttributeDict
import logging 
from wandb.integration.lightning.fabric import WandbLogger
import os

from lactchain.models.lightning_agent import LightningA2C
from lactchain.models.critic import ValueFunctionConfig
from lactchain.models.actor import ActorConfig, LoraConfigSettings
from lactchain.environments.grid_world import make_env, process_environment_outputs
from lactchain.configs.base_config import BaseConfig

PathLike=Union[str, Path]
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TORCH_LOGS']="+dynamo"
os.environ['TORCHDYNAMO_VERBOSE']='1'
ACTOR_PATH='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'
CRITIC_PATH='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11'

class FabricConfig(BaseConfig): 
    '''Base Config for running fabric accelerator'''
    accelerator:str=Field(
        default='gpu'
    )
    devices:int=Field(
        default=...
    )
    num_nodes:int=Field(
        default=1
    )
    strategy:str=Field(
        default='auto'
    )
    def __init__(self, devices:int): 
        super().__init__(devices=devices)

def configure_logger(level:str='debug') -> logging.Logger: 
    '''Function for creating a logger to write to file and terminal'''
    LEVELS={
        'debug':logging.DEBUG, 
        'info':logging.INFO, 
        'warning':logging.WARNING, 
        'error':logging.ERROR, 
        'critical':logging.CRITICAL
    }
    logger=logging.getLogger('Critic Training Logger')
    logger.setLevel(LEVELS.get(level))
    ch = logging.StreamHandler()
    ch.setLevel(LEVELS.get(level))
    fh = logging.FileHandler("training.log")
    fh.setLevel(LEVELS.get(level))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger

# def save_checkpoint(args:ArgumentParser, episode:int, rank:int, model:LightningA2C): 
#     if episdoe % args.checkpointing_steps == 0:
    #     if rank==0:
    #         # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    #         if args.checkpoints_total_limit is not None:
    #             checkpoints = os.listdir(args.output_dir)
    #             checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    #             checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    #             # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
    #             if len(checkpoints) >= args.checkpoints_total_limit:
    #                 num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
    #                 removing_checkpoints = checkpoints[0:num_to_remove]

    #                 logger.info(
    #                     f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
    #                 )
    #                 logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

    #                 for removing_checkpoint in removing_checkpoints:
    #                     removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
    #                     shutil.rmtree(removing_checkpoint)

    #         save_path = os.path.join(args.output_dir, f"actor-fined-tuned-critic-checkpoint-{episode}.pth")
    #         save_only_trainable_weights(model, save_path)
    #         logger.info(f"Saved state to {save_path}")


def save_only_trainable_weights(lightning_model:LightningA2C, path):
    '''Saves lora + q_value weights only'''
    torch.save(lightning_model.state_dict(), path)
    
def load_only_trainable_weights(lightning_model:LightningA2C, path):
    '''Loads lora + q_value weights only'''
    state_dict = torch.load(path)
    lightning_model.load_state_dict(state_dict, strict=False)

    
def unfold_list_of_lists(list_of_list:list[list[Any]]) -> list: 
    '''Unfolds a list of lists into a single large list that preserves order'''
    unfolded_list=list(itertools.chain.from_iterable(list_of_list))
    return unfolded_list

def train(
    fabric: Fabric,
    agent: LightningA2C,
    optimizer: torch.optim.Optimizer,
    lr_scheduler:torch.optim.lr_scheduler._LRScheduler, 
    data: Dict[str, Tensor],
    args: ArgumentParser,
    logger:Optional[logging.Logger]=None
):        
    rewards=torch.stack(unfold_list_of_lists(data['rewards']))
    observations=unfold_list_of_lists(data['observations'])
    infos=unfold_list_of_lists(data['infos'])
    logger.info(f'''TOTAL GATHERED DATA LENGTH: {len(rewards)}''')
    indexes=list(range(len(data['rewards'])))
    sampler = DistributedSampler(
        indexes, num_replicas=fabric.world_size, rank=fabric.global_rank, shuffle=True
    )
    sampler = BatchSampler(sampler, batch_size=args.per_rank_batch_size, drop_last=False)
    
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(args.update_epochs):
            epoch_loss=0
            sampler.sampler.set_epoch(epoch)
            for batch_indices in sampler:
                optimizer.zero_grad()
                selected_rewards=rewards[[batch_indices]]
                selected_obs=itemgetter(*batch_indices)(observations)
                selected_infos=itemgetter(*batch_indices)(infos)
                batch={'rewards':selected_rewards, 'observations':selected_obs, 'infos':selected_infos}
                loss = agent.training_step(batch) 
                epoch_loss+=loss
        
                fabric.backward(loss)
                optimizer.step()
                
            fabric.log_dict({"Epoch loss": epoch_loss})
            logger.info(f'Epoch Loss: {epoch_loss}')
            
            lr_scheduler.step()
        
def argparse(): 
    parser=ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--actor_path', type=str, default=ACTOR_PATH)
    parser.add_argument('--critic_path', type=str, default=CRITIC_PATH)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--buffer_size', type=int, default=100)
    parser.add_argument('--per_rank_batch_size', type=int, default=8)
    parser.add_argument('--update_epochs', type=int, default=10)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dtype', type=str, default='fp16')
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="actor-finetuned-critic",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    return parser.parse_args()

def main():
    logger = configure_logger()
    wandb_logger=WandbLogger(project="my-project", offline=True)
    # Initialize Fabric
    fabric = Fabric(loggers=wandb_logger)
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    
    args=argparse()
    actor_config=ActorConfig()
    lora_config=LoraConfigSettings()
    critic_config=ValueFunctionConfig()
    
    vector_env = gym.vector.AsyncVectorEnv([make_env for _ in range(args.num_envs)])
    
    # # Potentially load in the weights and states from a previous save
    # if args.resume_from_checkpoint:
    #     if args.resume_from_checkpoint != "latest":
    #         path = os.path.basename(args.resume_from_checkpoint)
    #     else:
    #         # Get the most recent checkpoint
    #         dirs = os.listdir(args.output_dir)
    #         dirs = [d for d in dirs if d.startswith("checkpoint")]
    #         dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    #         path = dirs[-1] if len(dirs) > 0 else None

    #     if path is None:
    #         fabric.print(
    #             f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
    #         )
    #         args.resume_from_checkpoint = None
    #     else:
    #         fabric.print(f"Resuming from checkpoint {path}")
    #         # fabric.load_state(os.path.join(args.output_dir, path))
    #         episode = int(path.split("-")[1])
    
    
    agent=LightningA2C(args.actor_path, actor_config, lora_config, 
                       args.critic_path, critic_config, args.gamma)
    logger.info(f'Model Trainable Params:\n{agent.model_trainable_params}')
    
    optimizer = agent.configure_optimizers(args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    agent, optimizer = fabric.setup(agent, optimizer)
    
    obs, info = vector_env.reset()
    obs, info=process_environment_outputs(obs, info)
    observations=[]
    rewards=[]
    infos=[]
    steps_kept=0
    with torch.autograd.set_detect_anomaly(True):
        with torch.no_grad(): 
            step=0
            # since we will all_gather to make buffer, desired buffer_size = world_size * per_rank_observation_counts
            while len(observations)*world_size<=args.buffer_size: 
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
                    pass
                
                logger.info(f'''Replay buffer size: {len(observations)} at step {step} for rank {rank}''')  
                
        
    logger.info(f'''FOR RANK {rank}\nTOTAL STEPS:{step}\nSTEPS KEPT:{steps_kept}\nGATHERING DATA FROM RANK {rank}...''')
    fabric.barrier()
    local_data={
        'rewards':[reward for reward in rewards], 
        'observations':[observation for observation in observations], 
        'infos':[info for info in infos]
        }
    gathered_data = fabric.all_gather(local_data)
    fabric.log_dict({'Total_reward':np.sum(np.stack(rewards))})
    
    train(fabric, agent, optimizer, lr_scheduler, gathered_data, args, logger)    
    
if __name__=="__main__": 
    
    main()